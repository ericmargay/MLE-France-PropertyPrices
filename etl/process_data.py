"""
Main ETL pipeline for French property prices analysis

This script:
1. Downloads raw data
2. Loads and cleans transaction data
3. Creates spatial geometries
4. Aggregates by 6 geographic levels
5. Saves to PostgreSQL/PostGIS
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry, WKTElement
from shapely.geometry import Point
import os
import sys
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from download_data import download_dvf_data, download_geometries, verify_downloads
from spatial_aggregation import (
    spatial_join_and_aggregate,
    aggregate_by_postcode,
    aggregate_by_parcel,
    create_country_aggregate,
    save_aggregates_to_postgres
)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
print(f"Connecting to database: {DATABASE_URL.split('@')[1]}")  # Hide password
engine = create_engine(DATABASE_URL)


def format_postcode(x):
    """
    Convert postcode to 5-digit string with leading zeros.
    Handles float values like 1630.0 -> '01630'
    
    French postcodes are always 5 digits, but pandas may read them as floats
    from CSV files, causing leading zeros to be lost.
    """
    if pd.isna(x) or str(x).strip() == '':
        return ''
    try:
        # Handle float values like 1630.0 -> 1630 -> '01630'
        num = int(float(x))
        return str(num).zfill(5)
    except (ValueError, TypeError):
        # Fallback: just strip and pad
        cleaned = str(x).strip()
        # Remove any decimal part if present
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        return cleaned.zfill(5)


def load_raw_transactions(filepath, sample_frac=None):
    """
    Load DVF transaction data from CSV
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file (can be gzipped)
    sample_frac : float, optional
        Fraction of data to sample (for testing, e.g., 0.1 = 10%)
    
    Returns:
    --------
    pd.DataFrame : Raw transaction data
    """
    print(f"\n{'='*60}")
    print(f"LOADING TRANSACTION DATA")
    print(f"{'='*60}\n")
    
    print(f"Reading from: {filepath}")
    print("⏳ This may take 5-10 minutes for large files...")
    print("   Loading and decompressing gzipped CSV...")
    
    start_time = time.time()
    
    # Read CSV (handles .gz automatically)
    df = pd.read_csv(
        filepath,
        low_memory=False,
        compression='gzip' if filepath.endswith('.gz') else None
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Loaded {len(df):,} rows in {elapsed:.1f} seconds")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Sample if requested (for testing)
    if sample_frac:
        original_len = len(df)
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Sampled {len(df):,} rows ({sample_frac*100}% of data)")
    
    return df


def clean_transaction_data(df):
    """
    Clean and filter transaction data
    
    Filters:
    - Residential properties only (Maison, Appartement)
    - Valid prices and surface areas
    - Remove outliers
    - Valid coordinates
    """
    print(f"\n{'='*60}")
    print(f"CLEANING TRANSACTION DATA")
    print(f"{'='*60}\n")
    
    original_len = len(df)
    
    # Step 1: Filter residential properties
    print("1. Filtering residential properties...")
    df = df[df['type_local'].isin(['Maison', 'Appartement'])]
    print(f"   Kept {len(df):,} residential transactions ({len(df)/original_len*100:.1f}%)")
    
    # Step 2: Remove rows with missing critical data
    print("2. Removing missing values...")
    required_columns = ['valeur_fonciere', 'surface_reelle_bati', 'longitude', 'latitude']
    df = df.dropna(subset=required_columns)
    print(f"   Kept {len(df):,} complete transactions ({len(df)/original_len*100:.1f}%)")
    
    # Step 3: Calculate price per m²
    print("3. Calculating price per m²...")
    df['price_per_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    
    # Step 4: Remove outliers
    print("4. Removing outliers...")
    
    # Price per m² filters (reasonable range for France)
    price_min = 500   # €/m²
    price_max = 20000  # €/m²
    
    # Surface area filters
    surface_min = 10   # m²
    surface_max = 500  # m²
    
    df = df[
        (df['price_per_m2'] >= price_min) & 
        (df['price_per_m2'] <= price_max) &
        (df['surface_reelle_bati'] >= surface_min) &
        (df['surface_reelle_bati'] <= surface_max)
    ]
    
    print(f"   Kept {len(df):,} valid transactions ({len(df)/original_len*100:.1f}%)")
    print(f"   Price range: €{df['price_per_m2'].min():.0f} - €{df['price_per_m2'].max():.0f}/m²")
    
    # Step 5: Convert date to datetime
    print("5. Converting dates...")
    df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce')
    df = df.dropna(subset=['date_mutation'])
    print(f"   Date range: {df['date_mutation'].min()} to {df['date_mutation'].max()}")
    
    # Step 6: Clean location codes - FIXED VERSION
    print("6. Cleaning location codes...")
    
    # CRITICAL FIX: Handle postcodes properly to preserve leading zeros
    # French postcodes like 01630 are read as floats (1630.0) by pandas
    # We need to convert them back to 5-digit strings with leading zeros
    df['code_postal'] = df['code_postal'].apply(format_postcode)
    df['code_commune'] = df['code_commune'].astype(str).str.strip()
    df['code_departement'] = df['code_departement'].astype(str).str.strip()
    
    # Verify postcode format
    sample_postcodes = df['code_postal'].head(10).tolist()
    print(f"   Sample postcodes: {sample_postcodes}")
    
    # Verify we have proper 5-digit postcodes
    valid_postcodes = df['code_postal'].str.len() == 5
    print(f"   Valid 5-digit postcodes: {valid_postcodes.sum():,} ({valid_postcodes.mean()*100:.1f}%)")
    
    # Step 7: Clean parcel IDs
    print("7. Cleaning parcel IDs...")
    if 'id_parcelle' in df.columns:
        df['id_parcelle'] = df['id_parcelle'].astype(str).str.strip()
        parcels_with_id = df['id_parcelle'].notna().sum()
        print(f"   Transactions with parcel ID: {parcels_with_id:,} ({parcels_with_id/len(df)*100:.1f}%)")
    
    print(f"\n✓ Final cleaned dataset: {len(df):,} transactions")
    
    return df


def create_geodataframe(df):
    """
    Convert DataFrame to GeoDataFrame with point geometries
    """
    print(f"\n{'='*60}")
    print(f"CREATING GEODATAFRAME")
    print(f"{'='*60}\n")
    
    print("Creating point geometries from coordinates...")
    
    start_time = time.time()
    
    # Create geometry from longitude/latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    elapsed = time.time() - start_time
    
    print(f"✓ Created GeoDataFrame with {len(gdf):,} points in {elapsed:.1f} seconds")
    print(f"  CRS: {gdf.crs}")
    print(f"  Bounds: {gdf.total_bounds}")
    
    return gdf


def load_to_postgres(gdf):
    """
    Load cleaned transaction data to PostgreSQL
    """
    print(f"\n{'='*60}")
    print(f"LOADING TO POSTGRESQL")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Prepare data for PostGIS
    gdf_copy = gdf.copy()
    
    # Convert geometry to WKT for PostGIS
    print("Converting geometries for PostGIS...")
    gdf_copy['geom'] = gdf_copy['geometry'].apply(
        lambda x: WKTElement(x.wkt, srid=4326)
    )
    
    # Select columns for database - INCLUDING PARCEL DATA
    columns = [
        'valeur_fonciere', 'date_mutation', 'type_local', 
        'surface_reelle_bati', 'code_commune', 'code_postal', 
        'code_departement', 'id_parcelle', 'surface_terrain',  # ← PARCEL COLUMNS
        'price_per_m2', 'geom'
    ]
    
    # Drop dependent views first
    print("Dropping dependent views if they exist...")
    with engine.connect() as conn:
        try:
            conn.execute(text("DROP VIEW IF EXISTS top_cities_by_volume CASCADE"))
            conn.execute(text("DROP VIEW IF EXISTS price_stats_summary CASCADE"))
            conn.commit()
            print("✓ Dropped existing views")
        except Exception as e:
            print(f"Note: {str(e)}")
            conn.rollback()
    
    # Write to PostgreSQL
    print("Writing to database...")
    write_start = time.time()
    
    gdf_copy[columns].to_sql(
        'transactions',
        engine,
        if_exists='replace',  # Replace existing data
        index=False,
        dtype={'geom': Geometry('POINT', srid=4326)},
        method='multi',
        chunksize=10000
    )
    
    write_time = time.time() - write_start
    print(f"✓ Loaded {len(gdf_copy):,} transactions to PostgreSQL in {write_time:.1f} seconds")
    
    # Recreate views
    print("Recreating views...")
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                CREATE OR REPLACE VIEW top_cities_by_volume AS
                SELECT 
                    code_commune,
                    type_local,
                    COUNT(*) as transaction_count,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_m2) as median_price,
                    AVG(price_per_m2) as avg_price,
                    STDDEV(price_per_m2) as std_dev
                FROM transactions
                WHERE price_per_m2 IS NOT NULL
                GROUP BY code_commune, type_local
                ORDER BY transaction_count DESC
            """))
            
            conn.execute(text("""
                CREATE OR REPLACE VIEW price_stats_summary AS
                SELECT 
                    level,
                    COUNT(*) as region_count,
                    AVG(median_price) as avg_median_price,
                    MIN(median_price) as min_price,
                    MAX(median_price) as max_price,
                    SUM(transaction_count) as total_transactions
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
            
            conn.commit()
            print("✓ Recreated views")
        except Exception as e:
            print(f"Note: Could not recreate views: {str(e)}")
            conn.rollback()
    
    # Verify
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM transactions"))
        count = result.scalar()
        print(f"✓ Verified: {count:,} rows in database")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Database loading completed in {elapsed:.1f} seconds")

        
def aggregate_all_levels(transactions_gdf):
    """
    Aggregate property prices by all 6 geographic levels
    
    Levels:
    1. Country (France)
    2. Region (13 regions)
    3. Département (96 départements)
    4. Commune/City (31,000+ communes) - serves as "Neighborhood"
    5. Postcode (code postal)
    6. Building Plots (parcelles cadastrales)
    """
    print(f"\n{'='*60}")
    print(f"STARTING SPATIAL AGGREGATION")
    print(f"{'='*60}")
    
    # Clear existing aggregates
    print("\nClearing existing aggregates...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE price_aggregates RESTART IDENTITY"))
        conn.commit()
    
    all_aggregates = []
    total_start = time.time()
    
    # Level 1: Country
    print(f"\n{'='*60}")
    print("LEVEL 1/6: COUNTRY")
    print(f"{'='*60}")
    level_start = time.time()
    
    country_agg = create_country_aggregate(transactions_gdf, country_code='FRA')
    all_aggregates.append(country_agg)
    save_aggregates_to_postgres(country_agg, engine, if_exists='append')
    
    level_time = time.time() - level_start
    print(f"\n✓ Country aggregation completed in {level_time:.1f} seconds")
    
    # Level 2: Regions
    print(f"\n{'='*60}")
    print("LEVEL 2/6: REGIONS")
    print(f"{'='*60}")
    level_start = time.time()
    
    print("Loading region boundaries...")
    regions = gpd.read_file('/app/data/geometries/regions.geojson')
    regions_agg = spatial_join_and_aggregate(
        transactions_gdf, 
        regions, 
        level_name='region',
        join_column='code',
        name_column='nom'
    )
    all_aggregates.append(regions_agg)
    save_aggregates_to_postgres(regions_agg, engine, if_exists='append')
    
    level_time = time.time() - level_start
    print(f"\n✓ Region aggregation completed in {level_time:.1f} seconds")
    
    # Level 3: Départements
    print(f"\n{'='*60}")
    print("LEVEL 3/6: DÉPARTEMENTS")
    print(f"{'='*60}")
    level_start = time.time()
    
    print("Loading département boundaries...")
    departements = gpd.read_file('/app/data/geometries/departements.geojson')
    departements_agg = spatial_join_and_aggregate(
        transactions_gdf,
        departements,
        level_name='departement',
        join_column='code',
        name_column='nom'
    )
    all_aggregates.append(departements_agg)
    save_aggregates_to_postgres(departements_agg, engine, if_exists='append')
    
    level_time = time.time() - level_start
    print(f"\n✓ Département aggregation completed in {level_time:.1f} seconds")
    
    # Level 4: Communes (serves as "Neighborhood")
    print(f"\n{'='*60}")
    print("LEVEL 4/6: COMMUNES (NEIGHBORHOODS)")
    print(f"{'='*60}")
    level_start = time.time()
    
    print("Loading commune boundaries...")
    communes = gpd.read_file('/app/data/geometries/communes.geojson')
    
    # Communes GeoJSON has different structure
    if 'code' in communes.columns:
        join_col = 'code'
    elif 'insee' in communes.columns:
        join_col = 'insee'
    else:
        print("Warning: Cannot find commune code column")
        join_col = communes.columns[0]
    
    communes_agg = spatial_join_and_aggregate(
        transactions_gdf,
        communes,
        level_name='commune',
        join_column=join_col,
        name_column='nom' if 'nom' in communes.columns else None
    )
    all_aggregates.append(communes_agg)
    save_aggregates_to_postgres(communes_agg, engine, if_exists='append')
    
    level_time = time.time() - level_start
    print(f"\n✓ Commune aggregation completed in {level_time:.1f} seconds")
    
    # Level 5: Postcodes
    print(f"\n{'='*60}")
    print("LEVEL 5/6: POSTCODES")
    print(f"{'='*60}")
    level_start = time.time()
    
    postcode_agg = aggregate_by_postcode(transactions_gdf, communes)
    all_aggregates.append(postcode_agg)
    save_aggregates_to_postgres(postcode_agg, engine, if_exists='append')
    
    level_time = time.time() - level_start
    print(f"\n✓ Postcode aggregation completed in {level_time:.1f} seconds")
    
    # Level 6: Building Plots (Parcels)
    print(f"\n{'='*60}")
    print("LEVEL 6/6: BUILDING PLOTS (PARCELS)")
    print(f"{'='*60}")
    level_start = time.time()
    
    parcel_agg = aggregate_by_parcel(transactions_gdf, max_parcels=300000)
    all_aggregates.append(parcel_agg)
    save_aggregates_to_postgres(parcel_agg, engine, if_exists='append')
    
    level_time = time.time() - level_start
    print(f"\n✓ Building plot aggregation completed in {level_time:.1f} seconds")
    
    # Summary
    total_time = time.time() - total_start
    total_records = sum(len(agg) for agg in all_aggregates)
    
    print(f"\n{'='*60}")
    print(f"AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Created {total_records:,} aggregated records across 6 levels")
    print(f"✓ Total aggregation time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"{'='*60}\n")
    
    return all_aggregates


def verify_aggregates():
    """
    Verify aggregated data in PostgreSQL
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING AGGREGATES")
    print(f"{'='*60}\n")
    
    with engine.connect() as conn:
        # Count by level
        result = conn.execute(text("""
            SELECT level, COUNT(*) as count, 
                   AVG(median_price) as avg_price,
                   SUM(transaction_count) as total_transactions
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
        
        print("Aggregation Summary:")
        print("-" * 70)
        for row in result:
            print(f"  {row.level:12} | {row.count:6,} regions | "
                  f"Avg: €{row.avg_price:6,.0f}/m² | "
                  f"Transactions: {row.total_transactions:,}")
        print("-" * 70)


def main():
    """
    Main ETL pipeline execution
    """
    overall_start = time.time()
    
    print("\n" + "="*60)
    print("FRANCE PROPERTY PRICES - ETL PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Step 1: Download data
        step_start = time.time()
        print("\n" + "="*60)
        print("STEP 1: DATA DOWNLOAD")
        print("="*60)
        
        if not verify_downloads():
            print("Downloading data...")
            download_dvf_data(2023)
            download_geometries()
        else:
            print("✓ Data already downloaded")
        
        step_time = time.time() - step_start
        print(f"\n✓ Step 1 completed in {step_time:.1f} seconds")
        
        # Step 2: Load and clean
        step_start = time.time()
        print("\n" + "="*60)
        print("STEP 2: DATA LOADING & CLEANING")
        print("="*60)
        
        df = load_raw_transactions(
            '/app/data/raw/dvf_2023.csv.gz',
            sample_frac=None  # Set to 0.1 for 10% sample
        )
        
        df_clean = clean_transaction_data(df)
        
        step_time = time.time() - step_start
        print(f"\n✓ Step 2 completed in {step_time/60:.1f} minutes ({step_time:.1f} seconds)")
        
        # Step 3: Create GeoDataFrame
        step_start = time.time()
        print("\n" + "="*60)
        print("STEP 3: SPATIAL PROCESSING")
        print("="*60)
        
        gdf = create_geodataframe(df_clean)
        
        step_time = time.time() - step_start
        print(f"\n✓ Step 3 completed in {step_time:.1f} seconds")
        
        # Step 4: Load to PostgreSQL
        step_start = time.time()
        print("\n" + "="*60)
        print("STEP 4: DATABASE LOADING")
        print("="*60)
        
        load_to_postgres(gdf)
        
        step_time = time.time() - step_start
        print(f"\n✓ Step 4 completed in {step_time/60:.1f} minutes ({step_time:.1f} seconds)")
        
        # Step 5: Spatial aggregation
        step_start = time.time()
        print("\n" + "="*60)
        print("STEP 5: SPATIAL AGGREGATION (6 LEVELS)")
        print("="*60)
        
        aggregates = aggregate_all_levels(gdf)
        
        step_time = time.time() - step_start
        print(f"✓ Step 5 completed in {step_time/60:.1f} minutes ({step_time:.1f} seconds)")
        
        # Step 6: Verify
        step_start = time.time()
        print("\n" + "="*60)
        print("STEP 6: VERIFICATION")
        print("="*60)
        
        verify_aggregates()
        
        step_time = time.time() - step_start
        print(f"\n✓ Step 6 completed in {step_time:.1f} seconds")
        
        # Final summary
        total_time = time.time() - overall_start
        
        print("\n" + "="*60)
        print("✓ ETL PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"Transactions loaded: {len(gdf):,}")
        print(f"Aggregates created: {sum(len(agg) for agg in aggregates):,}")
        print(f"Levels: Country, Region, Département, Commune, Postcode, Parcel")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
