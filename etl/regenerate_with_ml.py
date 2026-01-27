"""
Regenerate price aggregations using Robust ML Model.

This script implements hierarchical aggregation where:
1. Country is estimated first
2. Regions use country as prior
3. Departments use region as prior
4. Communes use department as prior
5. Postcodes use commune as prior
6. Parcels use commune as prior (with real building footprints)

This ensures estimates are smoothed appropriately for sparse areas.

Usage: docker-compose run --rm etl python regenerate_with_ml.py
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString
from shapely import affinity
from datetime import datetime
import os
import time
import warnings
warnings.filterwarnings('ignore')

from ml_price_model import RobustPriceModel, PREDICTION_DATE
from spatial_aggregation import (
    format_postcode,
    download_postcode_boundaries,
    save_aggregates_to_postgres
)
from cadastre_downloader import (
    download_all_buildings,
    load_cached_buildings,
    get_building_for_point,
    ALL_DEPARTMENTS
)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
engine = create_engine(DATABASE_URL)

# Configuration
CADASTRE_CACHE_DIR = '/app/data/cadastre'
MAX_PARCELS = 500000  # Limit for memory management


def load_transactions():
    """Load transactions from database."""
    print("\n" + "="*60)
    print("LOADING TRANSACTIONS")
    print("="*60)
    
    start = time.time()
    
    # Use PostGIS functions to extract coordinates from geometry column
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
    print(f"✓ Loaded {len(df):,} transactions in {time.time()-start:.1f}s")
    
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf


def aggregate_level(transactions_gdf, boundaries_gdf, level_name,
                    join_column, name_column, ml_model, parent_estimates=None):
    """
    Aggregate at a geographic level using ML with hierarchical smoothing.
    
    Parameters:
    -----------
    parent_estimates : dict
        Mapping from parent_code -> price estimate
        Used for hierarchical smoothing
    """
    print(f"\n{'='*60}")
    print(f"Aggregating: {level_name.upper()}")
    print(f"{'='*60}")
    
    if transactions_gdf.crs != boundaries_gdf.crs:
        transactions_gdf = transactions_gdf.to_crs(boundaries_gdf.crs)
    
    print("Performing spatial join...")
    joined = gpd.sjoin(transactions_gdf, boundaries_gdf, how='inner', predicate='within')
    
    print(f"Matched: {len(joined):,} transactions in {boundaries_gdf[join_column].nunique()} zones")
    
    aggregated_data = []
    codes = list(joined[join_column].unique())
    
    # Track estimates for child levels
    level_estimates = {}
    
    for i, code in enumerate(codes):
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1:,}/{len(codes):,} zones processed")
        
        group = joined[joined[join_column] == code]
        
        if len(group) == 0:
            continue
        
        # Get geometry
        geom_match = boundaries_gdf[boundaries_gdf[join_column] == code]
        if len(geom_match) == 0:
            continue
        geometry = geom_match.geometry.iloc[0]
        centroid = geometry.centroid
        
        # Get parent estimate for hierarchical smoothing
        parent_estimate = None
        if parent_estimates is not None:
            # Try to find parent code
            if 'code_departement' in group.columns and level_name == 'commune':
                parent_code = group['code_departement'].iloc[0]
                parent_estimate = parent_estimates.get(parent_code)
            elif 'code_region' in boundaries_gdf.columns:
                region_code = geom_match['code_region'].iloc[0] if 'code_region' in geom_match.columns else None
                if region_code:
                    parent_estimate = parent_estimates.get(region_code)
        
        # If no specific parent, use global median
        if parent_estimate is None:
            parent_estimate = ml_model.global_median_price
        
        # ML prediction with hierarchical smoothing
        stats = ml_model.predict_zone(
            group, 
            zone_centroid=(centroid.x, centroid.y),
            parent_estimate=parent_estimate
        )
        
        # Store estimate for child levels
        level_estimates[code] = stats['predicted_price']
        
        # Get name
        if name_column and name_column in geom_match.columns:
            name = geom_match[name_column].iloc[0]
        else:
            name = code
        
        aggregated_data.append({
            'level': level_name,
            'code': code,
            'name': name,
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=boundaries_gdf.crs)
    
    print(f"✓ Created {len(result_gdf):,} {level_name} zones")
    if len(result_gdf) > 0:
        print(f"  Price range: €{result_gdf['predicted_price'].min():,.0f} - €{result_gdf['predicted_price'].max():,.0f}/m²")
        print(f"  Avg confidence: {result_gdf['confidence_score'].mean():.0f}/100")
    
    return result_gdf, level_estimates


def aggregate_postcodes(transactions_gdf, communes_gdf, ml_model, commune_estimates):
    """Aggregate postcodes using official boundaries and hierarchical smoothing."""
    print(f"\n{'='*60}")
    print(f"Aggregating: POSTCODE")
    print(f"{'='*60}")
    
    # Download official boundaries
    postcode_boundaries = download_postcode_boundaries()
    
    if postcode_boundaries is not None and len(postcode_boundaries) > 0:
        print(f"  Using {len(postcode_boundaries):,} official postcode polygons")
        postcode_geom_lookup = {}
        for _, row in postcode_boundaries.iterrows():
            pc = format_postcode(row.get('code_postal', ''))
            if pc:
                postcode_geom_lookup[pc] = row.geometry
    else:
        print("  ⚠️  Official boundaries not available")
        postcode_geom_lookup = {}
    
    # Format postcodes
    print("Formatting postcodes...")
    transactions_gdf = transactions_gdf.copy()
    transactions_gdf['code_postal'] = transactions_gdf['code_postal'].apply(format_postcode)
    
    # Spatial join with communes to get commune codes
    print("Joining with communes...")
    if 'code' not in communes_gdf.columns and 'insee' in communes_gdf.columns:
        communes_gdf['code'] = communes_gdf['insee']
    
    joined = gpd.sjoin(
        transactions_gdf, 
        communes_gdf[['code', 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    # Group by postcode
    postcode_groups = joined.groupby('code_postal')
    unique_postcodes = [pc for pc in postcode_groups.groups.keys() if pc and pc != '']
    
    print(f"Processing {len(unique_postcodes):,} unique postcodes...")
    
    aggregated_data = []
    
    for i, postcode in enumerate(unique_postcodes):
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1:,}/{len(unique_postcodes):,} postcodes processed")
        
        group = postcode_groups.get_group(postcode)
        
        if len(group) == 0:
            continue
        
        # Get geometry
        if postcode in postcode_geom_lookup:
            geometry = postcode_geom_lookup[postcode]
        else:
            # Fallback: convex hull
            try:
                points = group.geometry.tolist()
                if len(points) >= 3:
                    geometry = MultiPoint(points).convex_hull.buffer(0.0005)
                elif len(points) == 2:
                    geometry = LineString(points).buffer(0.002)
                else:
                    geometry = points[0].buffer(0.003)
            except:
                continue
        
        if geometry is None or geometry.is_empty:
            continue
        
        # Get parent (commune) estimate
        commune_codes = group['code'].dropna().unique()
        parent_estimate = None
        if len(commune_codes) > 0:
            # Use median of commune estimates where this postcode has transactions
            parent_prices = [commune_estimates.get(c) for c in commune_codes if c in commune_estimates]
            if parent_prices:
                parent_estimate = np.median([p for p in parent_prices if p is not None])
        
        if parent_estimate is None:
            parent_estimate = ml_model.global_median_price
        
        # ML prediction
        centroid = geometry.centroid
        stats = ml_model.predict_zone(
            group,
            zone_centroid=(centroid.x, centroid.y),
            parent_estimate=parent_estimate
        )
        
        # Ensure valid polygon
        if isinstance(geometry, MultiPolygon):
            geometry = max(geometry.geoms, key=lambda g: g.area)
        
        aggregated_data.append({
            'level': 'postcode',
            'code': postcode,
            'name': f"{postcode}",
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=transactions_gdf.crs)
    
    print(f"\n✓ Created {len(result_gdf):,} postcode zones")
    if len(result_gdf) > 0:
        print(f"  Price range: €{result_gdf['predicted_price'].min():,.0f} - €{result_gdf['predicted_price'].max():,.0f}/m²")
    
    return result_gdf


def aggregate_parcels_ml(transactions_gdf, ml_model, commune_estimates, 
                         max_parcels=MAX_PARCELS):
    """
    Aggregate by parcel using cadastre building footprints and ML predictions.
    
    Downloads real building footprints from cadastre.data.gouv.fr and matches
    them to transaction locations.
    """
    print(f"\n{'='*60}")
    print(f"Aggregating: PARCEL (Building Footprints + ML)")
    print(f"{'='*60}")
    
    # Filter transactions with valid parcel IDs
    print("Filtering transactions with valid parcel IDs...")
    parcels_df = transactions_gdf[transactions_gdf['id_parcelle'].notna()].copy()
    parcels_df['id_parcelle'] = parcels_df['id_parcelle'].astype(str).str.strip()
    parcels_df = parcels_df[parcels_df['id_parcelle'] != '']
    parcels_df = parcels_df[parcels_df['id_parcelle'] != 'nan']
    
    print(f"  Transactions with parcel ID: {len(parcels_df):,}")
    
    # Extract department codes
    parcels_df['dept_code'] = parcels_df['id_parcelle'].str[:2]
    
    departments = parcels_df['dept_code'].unique()
    departments = [d for d in departments if d and len(d) >= 2]
    print(f"  Departments with parcels: {len(departments)}")
    
    # Group by parcel and limit if needed
    parcel_groups = parcels_df.groupby('id_parcelle')
    total_parcels = len(parcel_groups)
    print(f"  Unique parcels: {total_parcels:,}")
    
    if total_parcels > max_parcels:
        print(f"  ⚠️  Limiting to top {max_parcels:,} parcels by transaction count...")
        parcel_counts = parcel_groups.size().sort_values(ascending=False).head(max_parcels)
        selected_parcels = parcel_counts.index.tolist()
        parcels_df = parcels_df[parcels_df['id_parcelle'].isin(selected_parcels)]
        departments = parcels_df['dept_code'].unique()
        departments = [d for d in departments if d and len(d) >= 2]
        parcel_groups = parcels_df.groupby('id_parcelle')
    
    # Download cadastre building data
    print(f"\n" + "-"*60)
    print("DOWNLOADING CADASTRE BUILDING DATA")
    print("-"*60)
    
    # First try to load from cache
    cadastre_buildings = load_cached_buildings(CADASTRE_CACHE_DIR, departments)
    cached_count = len(cadastre_buildings)
    
    if cached_count < len(departments):
        # Download missing departments
        missing_depts = [d for d in departments if d not in cadastre_buildings]
        print(f"  Cached: {cached_count} departments")
        print(f"  Need to download: {len(missing_depts)} departments")
        
        new_buildings = download_all_buildings(
            departments=missing_depts,
            cache_dir=CADASTRE_CACHE_DIR,
            max_workers=4
        )
        
        cadastre_buildings.update(new_buildings)
    else:
        print(f"  ✓ All {len(departments)} departments loaded from cache")
    
    total_buildings = sum(len(gdf) for gdf in cadastre_buildings.values())
    print(f"\n  Total buildings loaded: {total_buildings:,}")
    
    # Process parcels
    print(f"\n" + "-"*60)
    print(f"PROCESSING {len(parcel_groups):,} PARCELS")
    print("-"*60)
    
    aggregated_data = []
    with_real_building = 0
    with_estimated = 0
    skipped = 0
    
    parcel_list = list(parcel_groups)
    
    for idx, (parcel_id, group) in enumerate(parcel_list):
        if (idx + 1) % 25000 == 0:
            pct = (idx + 1) / len(parcel_list) * 100
            print(f"  ... {idx + 1:,}/{len(parcel_list):,} ({pct:.1f}%) | "
                  f"Real: {with_real_building:,} | Est: {with_estimated:,}")
        
        if len(group) == 0:
            skipped += 1
            continue
        
        # Get department code
        dept = parcel_id[:2] if len(parcel_id) >= 2 else None
        
        # Try to find real building footprint
        geometry = None
        
        if dept and dept in cadastre_buildings:
            buildings_gdf = cadastre_buildings[dept]
            
            # Get transaction point
            tx_point = group.geometry.iloc[0]
            
            # Find building containing or near this point
            geometry = get_building_for_point(tx_point, buildings_gdf)
            
            if geometry is not None:
                with_real_building += 1
        
        # Fallback: create estimated footprint
        if geometry is None:
            try:
                if len(group) == 1:
                    point = group.geometry.iloc[0]
                    # Small rectangular building footprint (~12m x 12m)
                    geometry = point.buffer(0.00012, cap_style=3)
                else:
                    points = group.geometry.tolist()
                    multi_point = MultiPoint(points)
                    hull = multi_point.convex_hull
                    
                    if isinstance(hull, Point):
                        geometry = hull.buffer(0.00012, cap_style=3)
                    elif isinstance(hull, LineString):
                        geometry = hull.buffer(0.00006)
                    elif isinstance(hull, Polygon):
                        centroid = hull.centroid
                        geometry = affinity.scale(hull, xfact=1.15, yfact=1.15, origin=centroid)
                    else:
                        geometry = hull
                
                if isinstance(geometry, MultiPolygon):
                    geometry = max(geometry.geoms, key=lambda g: g.area)
                
                with_estimated += 1
                
            except Exception:
                skipped += 1
                continue
        
        if geometry is None or geometry.is_empty:
            skipped += 1
            continue
        
        # Get parent (commune) estimate for hierarchical smoothing
        commune_code = group['code_commune'].iloc[0] if 'code_commune' in group.columns else None
        parent_estimate = None
        
        if commune_code and commune_code in commune_estimates:
            parent_estimate = commune_estimates[commune_code]
        
        if parent_estimate is None:
            parent_estimate = ml_model.global_median_price
        
        # ML prediction
        centroid = geometry.centroid
        stats = ml_model.predict_zone(
            group,
            zone_centroid=(centroid.x, centroid.y),
            parent_estimate=parent_estimate
        )
        
        aggregated_data.append({
            'level': 'parcel',
            'code': parcel_id,
            'name': f"Parcel {parcel_id}",
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=transactions_gdf.crs)
    
    print(f"\n{'='*60}")
    print(f"✓ PARCEL AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total parcels: {len(result_gdf):,}")
    
    if len(result_gdf) > 0:
        pct_real = with_real_building / len(result_gdf) * 100
        print(f"  With REAL building footprints: {with_real_building:,} ({pct_real:.1f}%)")
        print(f"  With estimated footprints: {with_estimated:,} ({100-pct_real:.1f}%)")
        print(f"  Skipped: {skipped:,}")
        print(f"  Price range: €{result_gdf['predicted_price'].min():,.0f} - €{result_gdf['predicted_price'].max():,.0f}/m²")
        print(f"  Avg confidence: {result_gdf['confidence_score'].mean():.0f}/100")
    
    return result_gdf


def main():
    print("\n" + "="*70)
    print("   ROBUST ML PRICE AGGREGATION (WITH PARCELS)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Prediction target: {PREDICTION_DATE.strftime('%Y-%m-%d')}")
    
    start_time = time.time()
    
    # =================================================================
    # STEP 1: Load data
    # =================================================================
    transactions_gdf = load_transactions()
    
    # =================================================================
    # STEP 2: Train ML model
    # =================================================================
    print("\n" + "="*60)
    print("TRAINING ML MODEL")
    print("="*60)
    
    ml_model = RobustPriceModel(prediction_date=PREDICTION_DATE)
    ml_model.fit(transactions_gdf)
    
    # Save model
    model_path = '/app/data/ml_price_model.joblib'
    ml_model.save(model_path)
    
    # Print model summary
    print(ml_model.get_model_summary())
    
    # =================================================================
    # STEP 3: Clear existing aggregates
    # =================================================================
    print("\n" + "="*60)
    print("CLEARING EXISTING AGGREGATES")
    print("="*60)
    
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM price_aggregates"))
        conn.commit()
    print("✓ Cleared all existing aggregates")
    
    # =================================================================
    # STEP 4: Load boundaries
    # =================================================================
    print("\n" + "="*60)
    print("LOADING BOUNDARIES")
    print("="*60)
    
    geom_path = '/app/data/geometries'
    
    regions = gpd.read_file(f'{geom_path}/regions.geojson')
    print(f"  Regions: {len(regions)}")
    
    departements = gpd.read_file(f'{geom_path}/departements.geojson')
    print(f"  Départements: {len(departements)}")
    
    communes = gpd.read_file(f'{geom_path}/communes.geojson')
    print(f"  Communes: {len(communes)}")
    
    # =================================================================
    # STEP 5: Hierarchical aggregation
    # =================================================================
    
    # --- COUNTRY ---
    print("\n" + "="*60)
    print("Aggregating: COUNTRY")
    print("="*60)
    
    country_stats = ml_model.predict_zone(
        transactions_gdf,
        zone_centroid=(2.5, 46.6),
        parent_estimate=None
    )
    
    country_geometry = regions.geometry.unary_union
    
    country_gdf = gpd.GeoDataFrame([{
        'level': 'country',
        'code': 'FRA',
        'name': 'France',
        'geometry': country_geometry,
        **country_stats
    }], crs=transactions_gdf.crs)
    
    print(f"✓ France: €{country_stats['predicted_price']:,.0f}/m²")
    print(f"  Confidence: {country_stats['confidence_score']}/100")
    save_aggregates_to_postgres(country_gdf, engine, if_exists='append')
    
    country_estimate = {'FRA': country_stats['predicted_price']}
    
    # --- REGIONS ---
    region_gdf, region_estimates = aggregate_level(
        transactions_gdf, regions, 'region', 'code', 'nom',
        ml_model, parent_estimates=country_estimate
    )
    save_aggregates_to_postgres(region_gdf, engine, if_exists='append')
    
    # --- DEPARTMENTS ---
    # Map departments to regions for parent estimates
    dept_region_map = {}
    if 'code_region' in departements.columns:
        dept_region_map = departements.set_index('code')['code_region'].to_dict()
    
    dept_parent_estimates = {}
    for dept_code, region_code in dept_region_map.items():
        if region_code in region_estimates:
            dept_parent_estimates[dept_code] = region_estimates[region_code]
    
    dept_gdf, dept_estimates = aggregate_level(
        transactions_gdf, departements, 'departement', 'code', 'nom',
        ml_model, parent_estimates=dept_parent_estimates if dept_parent_estimates else region_estimates
    )
    save_aggregates_to_postgres(dept_gdf, engine, if_exists='append')
    
    # --- COMMUNES ---
    # Map communes to departments
    commune_dept_map = {}
    if 'code' in communes.columns:
        commune_dept_map = {row['code']: row['code'][:2] for _, row in communes.iterrows()}
    
    commune_parent_estimates = {}
    for commune_code, dept_code in commune_dept_map.items():
        if dept_code in dept_estimates:
            commune_parent_estimates[commune_code] = dept_estimates[dept_code]
    
    commune_gdf, commune_estimates = aggregate_level(
        transactions_gdf, communes, 'commune', 'code', 'nom',
        ml_model, parent_estimates=commune_parent_estimates if commune_parent_estimates else dept_estimates
    )
    save_aggregates_to_postgres(commune_gdf, engine, if_exists='append')
    
    # --- POSTCODES ---
    postcode_gdf = aggregate_postcodes(
        transactions_gdf, communes, ml_model, commune_estimates
    )
    save_aggregates_to_postgres(postcode_gdf, engine, if_exists='append')
    
    # --- PARCELS ---
    parcel_gdf = aggregate_parcels_ml(
        transactions_gdf, ml_model, commune_estimates,
        max_parcels=MAX_PARCELS
    )
    save_aggregates_to_postgres(parcel_gdf, engine, if_exists='append')
    
    # =================================================================
    # VERIFICATION
    # =================================================================
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT level, 
                   COUNT(*) as zones,
                   ROUND(AVG(median_price)) as avg_price,
                   ROUND(AVG(confidence_score)) as avg_conf,
                   SUM(transaction_count) as total_tx
            FROM price_aggregates 
            GROUP BY level
            ORDER BY 
                CASE level
                    WHEN 'country' THEN 1
                    WHEN 'region' THEN 2
                    WHEN 'departement' THEN 3
                    WHEN 'commune' THEN 4
                    WHEN 'postcode' THEN 5
                    WHEN 'parcel' THEN 6
                END
        """))
        
        print("\n" + "-"*70)
        print(f"{'Level':<12} | {'Zones':>8} | {'Avg Price':>12} | {'Confidence':>10} | {'Transactions':>12}")
        print("-"*70)
        for row in result:
            print(f"{row[0]:<12} | {row[1]:>8,} | €{row[2]:>10,}/m² | {row[3]:>8}/100 | {row[4]:>12,}")
        print("-"*70)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✓ ML AGGREGATION COMPLETE (ALL LEVELS)")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Model saved: {model_path}")
    print(f"\nModel metrics:")
    print(f"  - CV R²: {ml_model.cv_r2:.3f}")
    print(f"  - Annual trend: {(np.exp(ml_model.annual_trend)-1)*100:+.2f}%")
    print("\nRestart webapp: docker-compose restart webapp")
    print("="*70)


if __name__ == "__main__":
    main()
