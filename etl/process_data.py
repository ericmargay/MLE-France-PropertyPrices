"""
Main ETL Pipeline for France Property Price Analysis
=====================================================

This script processes DVF (Demandes de Valeurs Foncières) data and creates
price aggregations using Machine Learning.

PIPELINE STEPS:
1. Download DVF data from data.gouv.fr
2. Download geographic boundaries (regions, departments, communes)
3. Clean and transform transaction data
4. Train ML model on transactions
5. Aggregate prices at multiple geographic levels using ML
6. Save to PostgreSQL with PostGIS geometries

Usage:
    docker-compose run --rm etl python process_data.py

Environment Variables:
    DATABASE_URL: PostgreSQL connection string
    DVF_YEARS: Comma-separated years to process (default: 2023)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry, WKTElement
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from datetime import datetime
import requests
import os
import gzip
import time
import warnings
warnings.filterwarnings('ignore')

# Import ML model and spatial aggregation functions
from ml_price_model import RobustPriceModel, PREDICTION_DATE
from spatial_aggregation import (
    format_postcode,
    download_postcode_boundaries,
    save_aggregates_to_postgres,
    download_cadastre_batiments
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')

DVF_YEARS = os.getenv('DVF_YEARS', '2023').split(',')

DATA_DIR = '/app/data'
GEOMETRY_DIR = f'{DATA_DIR}/geometries'

# DVF data URLs
DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"

# Geographic boundary URLs
GEO_URLS = {
    'regions': 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson',
    'departements': 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson',
    'communes': 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes-version-simplifiee.geojson'
}


def setup_database(engine):
    """Create necessary database tables and extensions."""
    print("\n" + "="*60)
    print("SETTING UP DATABASE")
    print("="*60)
    
    with engine.connect() as conn:
        # Enable PostGIS
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.commit()
        print("✓ PostGIS extension enabled")
        
        # Create transactions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                date_mutation DATE,
                nature_mutation VARCHAR(50),
                valeur_fonciere DECIMAL(15,2),
                code_postal VARCHAR(10),
                code_commune VARCHAR(10),
                code_departement VARCHAR(5),
                type_local VARCHAR(50),
                surface_reelle_bati DECIMAL(10,2),
                nombre_pieces_principales INTEGER,
                id_parcelle VARCHAR(50),
                price_per_m2 DECIMAL(10,2),
                geom GEOMETRY(Point, 4326)
            )
        """))
        conn.commit()
        print("✓ Transactions table created")
        
        # Create aggregates table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS price_aggregates (
                id SERIAL PRIMARY KEY,
                level VARCHAR(20),
                code VARCHAR(50),
                name VARCHAR(255),
                median_price DECIMAL(10,2),
                weighted_price DECIMAL(10,2),
                std_dev DECIMAL(10,2),
                transaction_count INTEGER,
                lower_bound DECIMAL(10,2),
                upper_bound DECIMAL(10,2),
                last_transaction_date DATE,
                confidence_score INTEGER,
                estimate_quality VARCHAR(20),
                geom GEOMETRY(MULTIPOLYGON, 4326)
            )
        """))
        conn.commit()
        print("✓ Price aggregates table created")
        
        # Create spatial indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_transactions_geom 
            ON transactions USING GIST (geom)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_aggregates_geom 
            ON price_aggregates USING GIST (geom)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_aggregates_level 
            ON price_aggregates (level)
        """))
        conn.commit()
        print("✓ Spatial indexes created")


def download_dvf_data(year):
    """Download DVF data for a specific year."""
    print(f"\nDownloading DVF data for {year}...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    url = f"{DVF_BASE_URL}/{year}/full.csv.gz"
    local_path = f"{DATA_DIR}/dvf_{year}.csv.gz"
    
    if os.path.exists(local_path):
        print(f"  Using cached file: {local_path}")
        return local_path
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print(f"  ✓ Downloaded: {local_path}")
        return local_path
    else:
        print(f"  ✗ Failed to download: HTTP {response.status_code}")
        return None


def download_geometries():
    """Download geographic boundary files."""
    print("\n" + "="*60)
    print("DOWNLOADING GEOGRAPHIC BOUNDARIES")
    print("="*60)
    
    os.makedirs(GEOMETRY_DIR, exist_ok=True)
    
    for name, url in GEO_URLS.items():
        local_path = f"{GEOMETRY_DIR}/{name}.geojson"
        
        if os.path.exists(local_path):
            print(f"  Using cached: {name}")
            continue
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(local_path, 'w') as f:
                    f.write(response.text)
                print(f"  ✓ Downloaded: {name}")
            else:
                print(f"  ✗ Failed: {name}")
        except Exception as e:
            print(f"  ✗ Error downloading {name}: {e}")


def load_and_clean_dvf(year):
    """Load and clean DVF data for a year."""
    print(f"\nProcessing DVF data for {year}...")
    
    file_path = f"{DATA_DIR}/dvf_{year}.csv.gz"
    
    if not os.path.exists(file_path):
        file_path = download_dvf_data(year)
        if file_path is None:
            return None
    
    # Load with specific columns
    columns_to_load = [
        'date_mutation', 'nature_mutation', 'valeur_fonciere',
        'code_postal', 'code_commune', 'code_departement',
        'type_local', 'surface_reelle_bati', 'nombre_pieces_principales',
        'id_parcelle', 'longitude', 'latitude'
    ]
    
    print("  Loading CSV...")
    df = pd.read_csv(
        file_path, 
        compression='gzip',
        usecols=columns_to_load,
        low_memory=False
    )
    
    print(f"  Raw rows: {len(df):,}")
    
    # Filter to property sales with valid data
    df = df[df['nature_mutation'] == 'Vente']
    df = df[df['type_local'].isin(['Maison', 'Appartement'])]
    df = df[df['valeur_fonciere'].notna() & (df['valeur_fonciere'] > 0)]
    df = df[df['surface_reelle_bati'].notna() & (df['surface_reelle_bati'] > 0)]
    df = df[df['longitude'].notna() & df['latitude'].notna()]
    
    # Calculate price per m²
    df['price_per_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    
    # Filter outliers
    df = df[(df['price_per_m2'] > 100) & (df['price_per_m2'] < 50000)]
    df = df[(df['surface_reelle_bati'] >= 9) & (df['surface_reelle_bati'] <= 500)]
    
    # Clean postcodes
    df['code_postal'] = df['code_postal'].apply(format_postcode)
    
    # Ensure department code
    if 'code_departement' not in df.columns or df['code_departement'].isna().any():
        df['code_departement'] = df['code_commune'].astype(str).str[:2]
    
    print(f"  Cleaned rows: {len(df):,}")
    
    return df


def save_transactions_to_db(df, engine):
    """Save transaction data to PostgreSQL."""
    print("\nSaving transactions to database...")
    
    # Create geometry column
    df['geom'] = df.apply(
        lambda row: WKTElement(
            Point(row['longitude'], row['latitude']).wkt, 
            srid=4326
        ),
        axis=1
    )
    
    # Select columns for database
    columns = [
        'date_mutation', 'nature_mutation', 'valeur_fonciere',
        'code_postal', 'code_commune', 'code_departement',
        'type_local', 'surface_reelle_bati', 'nombre_pieces_principales',
        'id_parcelle', 'price_per_m2', 'geom'
    ]
    
    df_to_save = df[columns].copy()
    
    # Save in chunks
    chunk_size = 50000
    total_chunks = (len(df_to_save) // chunk_size) + 1
    
    for i in range(0, len(df_to_save), chunk_size):
        chunk = df_to_save.iloc[i:i+chunk_size]
        chunk.to_sql(
            'transactions',
            engine,
            if_exists='append',
            index=False,
            dtype={'geom': Geometry('POINT', srid=4326)}
        )
        print(f"  Saved chunk {i//chunk_size + 1}/{total_chunks}")
    
    print(f"✓ Saved {len(df_to_save):,} transactions")


def load_transactions_as_gdf(engine):
    """Load transactions from database as GeoDataFrame."""
    print("\nLoading transactions from database...")
    
    query = """
        SELECT 
            valeur_fonciere,
            date_mutation,
            type_local,
            surface_reelle_bati,
            code_commune,
            code_postal,
            code_departement,
            id_parcelle,
            price_per_m2,
            ST_X(geom) as longitude,
            ST_Y(geom) as latitude
        FROM transactions
        WHERE price_per_m2 IS NOT NULL 
          AND price_per_m2 > 100
          AND price_per_m2 < 50000
          AND geom IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print(f"✓ Loaded {len(gdf):,} transactions")
    return gdf


def aggregate_with_ml(transactions_gdf, boundaries_gdf, level_name,
                      join_column, name_column, ml_model, parent_estimates=None):
    """Aggregate at a geographic level using ML."""
    print(f"\n{'='*60}")
    print(f"Aggregating: {level_name.upper()}")
    print(f"{'='*60}")
    
    if transactions_gdf.crs != boundaries_gdf.crs:
        transactions_gdf = transactions_gdf.to_crs(boundaries_gdf.crs)
    
    joined = gpd.sjoin(transactions_gdf, boundaries_gdf, how='inner', predicate='within')
    print(f"Matched: {len(joined):,} transactions")
    
    aggregated_data = []
    codes = list(joined[join_column].unique())
    level_estimates = {}
    
    for i, code in enumerate(codes):
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1:,}/{len(codes):,} zones")
        
        group = joined[joined[join_column] == code]
        if len(group) == 0:
            continue
        
        geom_match = boundaries_gdf[boundaries_gdf[join_column] == code]
        if len(geom_match) == 0:
            continue
        
        geometry = geom_match.geometry.iloc[0]
        centroid = geometry.centroid
        
        # Get parent estimate
        parent_estimate = None
        if parent_estimates:
            if 'code_departement' in group.columns and level_name == 'commune':
                parent_code = group['code_departement'].iloc[0]
                parent_estimate = parent_estimates.get(parent_code)
        
        if parent_estimate is None:
            parent_estimate = ml_model.global_median_price
        
        # ML prediction
        stats = ml_model.predict_zone(
            group,
            zone_centroid=(centroid.x, centroid.y),
            parent_estimate=parent_estimate
        )
        
        level_estimates[code] = stats['predicted_price']
        
        name = geom_match[name_column].iloc[0] if name_column in geom_match.columns else code
        
        aggregated_data.append({
            'level': level_name,
            'code': code,
            'name': name,
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=boundaries_gdf.crs)
    
    print(f"✓ Created {len(result_gdf):,} zones")
    if len(result_gdf) > 0:
        print(f"  Price range: €{result_gdf['predicted_price'].min():,.0f} - €{result_gdf['predicted_price'].max():,.0f}/m²")
    
    return result_gdf, level_estimates


def aggregate_postcodes_ml(transactions_gdf, communes_gdf, ml_model, commune_estimates):
    """Aggregate postcodes using official boundaries and ML."""
    print(f"\n{'='*60}")
    print(f"Aggregating: POSTCODE")
    print(f"{'='*60}")
    
    postcode_boundaries = download_postcode_boundaries()
    
    postcode_geom_lookup = {}
    if postcode_boundaries is not None:
        for _, row in postcode_boundaries.iterrows():
            pc = format_postcode(row.get('code_postal', ''))
            if pc:
                postcode_geom_lookup[pc] = row.geometry
    
    transactions_gdf = transactions_gdf.copy()
    transactions_gdf['code_postal'] = transactions_gdf['code_postal'].apply(format_postcode)
    
    # Join with communes
    if 'code' not in communes_gdf.columns and 'insee' in communes_gdf.columns:
        communes_gdf['code'] = communes_gdf['insee']
    
    joined = gpd.sjoin(
        transactions_gdf,
        communes_gdf[['code', 'geometry']],
        how='left',
        predicate='within'
    )
    
    postcode_groups = joined.groupby('code_postal')
    unique_postcodes = [pc for pc in postcode_groups.groups.keys() if pc]
    
    print(f"Processing {len(unique_postcodes):,} postcodes...")
    
    aggregated_data = []
    
    for i, postcode in enumerate(unique_postcodes):
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1:,}/{len(unique_postcodes):,}")
        
        group = postcode_groups.get_group(postcode)
        if len(group) == 0:
            continue
        
        # Get geometry
        if postcode in postcode_geom_lookup:
            geometry = postcode_geom_lookup[postcode]
        else:
            try:
                points = group.geometry.tolist()
                if len(points) >= 3:
                    geometry = MultiPoint(points).convex_hull.buffer(0.0005)
                else:
                    geometry = points[0].buffer(0.003)
            except:
                continue
        
        if geometry is None or geometry.is_empty:
            continue
        
        # Parent estimate
        commune_codes = group['code'].dropna().unique()
        parent_prices = [commune_estimates.get(c) for c in commune_codes if c in commune_estimates]
        parent_estimate = np.median([p for p in parent_prices if p]) if parent_prices else ml_model.global_median_price
        
        centroid = geometry.centroid
        stats = ml_model.predict_zone(
            group,
            zone_centroid=(centroid.x, centroid.y),
            parent_estimate=parent_estimate
        )
        
        if isinstance(geometry, MultiPolygon):
            geometry = max(geometry.geoms, key=lambda g: g.area)
        
        aggregated_data.append({
            'level': 'postcode',
            'code': postcode,
            'name': postcode,
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=transactions_gdf.crs)
    
    print(f"✓ Created {len(result_gdf):,} postcode zones")
    return result_gdf


def main():
    """Main ETL pipeline."""
    print("\n" + "="*70)
    print("   FRANCE PROPERTY PRICE ETL PIPELINE (ML)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Years to process: {DVF_YEARS}")
    print(f"Prediction target: {PREDICTION_DATE.strftime('%Y-%m-%d')}")
    
    start_time = time.time()
    
    # Database connection
    engine = create_engine(DATABASE_URL)
    print(f"\nDatabase: {DATABASE_URL.split('@')[1]}")
    
    # =================================================================
    # STEP 1: Setup
    # =================================================================
    setup_database(engine)
    download_geometries()
    
    # =================================================================
    # STEP 2: Load and process DVF data
    # =================================================================
    print("\n" + "="*60)
    print("PROCESSING DVF DATA")
    print("="*60)
    
    # Clear existing transactions
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE transactions"))
        conn.commit()
    
    for year in DVF_YEARS:
        df = load_and_clean_dvf(year.strip())
        if df is not None and len(df) > 0:
            save_transactions_to_db(df, engine)
    
    # =================================================================
    # STEP 3: Load transactions as GeoDataFrame
    # =================================================================
    transactions_gdf = load_transactions_as_gdf(engine)
    
    # =================================================================
    # STEP 4: Train ML model
    # =================================================================
    print("\n" + "="*60)
    print("TRAINING ML MODEL")
    print("="*60)
    
    ml_model = RobustPriceModel(prediction_date=PREDICTION_DATE)
    ml_model.fit(transactions_gdf)
    
    # Save model
    model_path = f'{DATA_DIR}/ml_price_model.joblib'
    ml_model.save(model_path)
    
    # =================================================================
    # STEP 5: Clear and regenerate aggregates
    # =================================================================
    print("\n" + "="*60)
    print("GENERATING PRICE AGGREGATES (ML)")
    print("="*60)
    
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE price_aggregates"))
        conn.commit()
    
    # Load boundaries
    regions = gpd.read_file(f'{GEOMETRY_DIR}/regions.geojson')
    departements = gpd.read_file(f'{GEOMETRY_DIR}/departements.geojson')
    communes = gpd.read_file(f'{GEOMETRY_DIR}/communes.geojson')
    
    # Country
    print("\n" + "="*60)
    print("Aggregating: COUNTRY")
    print("="*60)
    
    country_stats = ml_model.predict_zone(
        transactions_gdf,
        zone_centroid=(2.5, 46.6)
    )
    
    country_gdf = gpd.GeoDataFrame([{
        'level': 'country',
        'code': 'FRA',
        'name': 'France',
        'geometry': regions.geometry.unary_union,
        **country_stats
    }], crs=transactions_gdf.crs)
    
    print(f"✓ France: €{country_stats['predicted_price']:,.0f}/m²")
    save_aggregates_to_postgres(country_gdf, engine, if_exists='append')
    
    country_estimate = {'FRA': country_stats['predicted_price']}
    
    # Regions
    region_gdf, region_estimates = aggregate_with_ml(
        transactions_gdf, regions, 'region', 'code', 'nom',
        ml_model, parent_estimates=country_estimate
    )
    save_aggregates_to_postgres(region_gdf, engine, if_exists='append')
    
    # Departments
    dept_gdf, dept_estimates = aggregate_with_ml(
        transactions_gdf, departements, 'departement', 'code', 'nom',
        ml_model, parent_estimates=region_estimates
    )
    save_aggregates_to_postgres(dept_gdf, engine, if_exists='append')
    
    # Communes
    commune_parent = {row['code']: dept_estimates.get(row['code'][:2]) for _, row in communes.iterrows()}
    
    commune_gdf, commune_estimates = aggregate_with_ml(
        transactions_gdf, communes, 'commune', 'code', 'nom',
        ml_model, parent_estimates=commune_parent
    )
    save_aggregates_to_postgres(commune_gdf, engine, if_exists='append')
    
    # Postcodes
    postcode_gdf = aggregate_postcodes_ml(
        transactions_gdf, communes, ml_model, commune_estimates
    )
    save_aggregates_to_postgres(postcode_gdf, engine, if_exists='append')
    
    # =================================================================
    # VERIFICATION
    # =================================================================
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT level, COUNT(*) as zones,
                   ROUND(AVG(median_price)) as avg_price,
                   ROUND(AVG(confidence_score)) as avg_conf
            FROM price_aggregates
            GROUP BY level
            ORDER BY 
                CASE level
                    WHEN 'country' THEN 1
                    WHEN 'region' THEN 2
                    WHEN 'departement' THEN 3
                    WHEN 'commune' THEN 4
                    WHEN 'postcode' THEN 5
                END
        """))
        
        print("\n" + "-"*60)
        print(f"{'Level':<12} | {'Zones':>8} | {'Avg Price':>12} | {'Confidence':>10}")
        print("-"*60)
        for row in result:
            print(f"{row[0]:<12} | {row[1]:>8,} | €{row[2]:>10,}/m² | {row[3]:>8}/100")
        print("-"*60)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✓ ETL PIPELINE COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Model R²: {ml_model.cv_r2:.3f}")
    print(f"Annual trend: {(np.exp(ml_model.annual_trend)-1)*100:+.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
