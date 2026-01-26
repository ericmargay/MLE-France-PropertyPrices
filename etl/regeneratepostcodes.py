"""
Script to regenerate PARCEL aggregations with 50% real cadastre buildings.
For testing purposes - downloads partial cadastre data to avoid timeouts.

Usage: docker-compose run --rm etl python regenerate_parcels_test.py
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
import requests
import gzip
import json
import warnings
warnings.filterwarnings('ignore')

from spatial_aggregation import (
    format_postcode,
    calculate_price_statistics,
    save_aggregates_to_postgres
)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
print(f"Connecting to database: {DATABASE_URL.split('@')[1]}")
engine = create_engine(DATABASE_URL)


def load_transactions_from_db():
    """Load transaction data from PostgreSQL into GeoDataFrame"""
    print("\n" + "="*60)
    print("LOADING TRANSACTIONS FROM DATABASE")
    print("="*60)
    
    start = time.time()
    
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
    """
    
    df = pd.read_sql(query, engine)
    print(f"✓ Loaded {len(df):,} transactions in {time.time()-start:.1f}s")
    
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf


def download_cadastre_batiments_sampled(departement_code, sample_pct=0.5, cache_dir='/app/data/cadastre'):
    """
    Download cadastre BATIMENTS (building footprints) for a department.
    Only keeps a random sample to reduce memory and processing time.
    
    Parameters:
    -----------
    sample_pct : float
        Percentage of buildings to keep (0.5 = 50%)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    dept = str(departement_code).zfill(2) if str(departement_code).isdigit() else str(departement_code)
    
    # Check for cached sampled file
    cache_file = os.path.join(cache_dir, f'batiments_{dept}_sampled.geojson')
    failed_marker = os.path.join(cache_dir, f'batiments_{dept}.failed')
    
    # Skip if previously failed
    if os.path.exists(failed_marker):
        return None
    
    # Use cached sampled file if exists
    if os.path.exists(cache_file):
        try:
            return gpd.read_file(cache_file)
        except Exception:
            pass
    
    # Skip very large departments entirely for this test
    skip_depts = ['75', '92', '93', '94']  # Paris and close suburbs - huge files
    if dept in skip_depts:
        print(f"      Skipping dept {dept} (too large for test)")
        return None
    
    url = f"https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/departements/{dept}/cadastre-{dept}-batiments.json.gz"
    
    try:
        print(f"      Downloading dept {dept}...", end=" ", flush=True)
        
        # Quick timeout - if it takes too long, skip
        response = requests.get(url, timeout=30, stream=True)
        
        if response.status_code != 200:
            print(f"HTTP {response.status_code}")
            return None
        
        # Check size - skip if > 50MB compressed
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 50 * 1024 * 1024:
            print(f"too large ({int(content_length)/1024/1024:.0f}MB)")
            with open(failed_marker, 'w') as f:
                f.write('too large')
            return None
        
        # Download with size limit
        chunks = []
        downloaded = 0
        max_size = 50 * 1024 * 1024  # 50MB max
        
        for chunk in response.iter_content(chunk_size=512*1024):
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded > max_size:
                print(f"exceeded {max_size/1024/1024:.0f}MB limit")
                with open(failed_marker, 'w') as f:
                    f.write('exceeded size')
                return None
        
        content = b''.join(chunks)
        
        # Decompress
        try:
            decompressed = gzip.decompress(content)
        except Exception as e:
            print(f"decompress failed")
            return None
        
        # Parse JSON
        try:
            geojson_data = json.loads(decompressed)
        except Exception as e:
            print(f"JSON parse failed")
            return None
        
        features = geojson_data.get('features', [])
        if len(features) == 0:
            print(f"no features")
            return None
        
        # SAMPLE: Keep only sample_pct of features randomly
        n_total = len(features)
        n_keep = int(n_total * sample_pct)
        
        np.random.seed(42)  # Reproducible sampling
        indices = np.random.choice(n_total, size=n_keep, replace=False)
        sampled_features = [features[i] for i in indices]
        
        # Create GeoDataFrame from sampled features
        gdf = gpd.GeoDataFrame.from_features(sampled_features, crs="EPSG:4326")
        
        # Save sampled version to cache
        try:
            gdf.to_file(cache_file, driver='GeoJSON')
        except:
            pass
        
        print(f"✓ {n_keep:,}/{n_total:,} buildings ({sample_pct*100:.0f}%)")
        return gdf
        
    except requests.exceptions.Timeout:
        print(f"timeout")
        with open(failed_marker, 'w') as f:
            f.write('timeout')
        return None
    except Exception as e:
        print(f"error: {e}")
        return None


def aggregate_by_parcel_test(transactions_gdf, max_parcels=200000, sample_pct=0.5):
    """
    Aggregate property prices by cadastral parcel.
    TEST VERSION: Uses 50% real cadastre buildings, 50% rectangles.
    """
    print(f"\n{'='*60}")
    print(f"Aggregating by BUILDING PLOT (TEST: {sample_pct*100:.0f}% Real Buildings)")
    print(f"{'='*60}")
    
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
    
    # Group by parcel first to limit
    parcel_groups = parcels_df.groupby('id_parcelle')
    total_parcels = len(parcel_groups)
    print(f"  Unique parcels: {total_parcels:,}")
    
    if total_parcels > max_parcels:
        print(f"  ⚠️  Limiting to {max_parcels:,} parcels for testing...")
        parcel_counts = parcel_groups.size().sort_values(ascending=False).head(max_parcels)
        selected_parcels = parcel_counts.index.tolist()
        parcels_df = parcels_df[parcels_df['id_parcelle'].isin(selected_parcels)]
        
        # Recalculate departments after filtering
        departments = parcels_df['dept_code'].unique()
        departments = [d for d in departments if d and len(d) >= 2]
        parcel_groups = parcels_df.groupby('id_parcelle')
    
    # Download sampled cadastre data
    print(f"\nDownloading cadastre (sampled {sample_pct*100:.0f}%) for {len(departments)} departments...")
    cadastre_buildings = {}
    
    for i, dept in enumerate(sorted(departments)):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(departments)} departments")
        
        try:
            buildings_gdf = download_cadastre_batiments_sampled(dept, sample_pct=sample_pct)
            if buildings_gdf is not None and len(buildings_gdf) > 0:
                cadastre_buildings[dept] = buildings_gdf
        except Exception as e:
            continue
    
    print(f"\n  ✓ Loaded cadastre for {len(cadastre_buildings)}/{len(departments)} departments")
    total_buildings = sum(len(gdf) for gdf in cadastre_buildings.values())
    print(f"  ✓ Total buildings in cache: {total_buildings:,}")
    
    # Process parcels
    print(f"\nProcessing {len(parcel_groups):,} parcels...")
    aggregated_data = []
    with_real_building = 0
    with_rectangle = 0
    skipped = 0
    
    for idx, (parcel_id, group) in enumerate(parcel_groups):
        if (idx + 1) % 25000 == 0:
            print(f"  ... {idx + 1:,}/{len(parcel_groups):,} parcels "
                  f"(real: {with_real_building:,}, rect: {with_rectangle:,})")
        
        if len(group) == 0:
            skipped += 1
            continue
        
        stats = calculate_price_statistics(group[['price_per_m2', 'date_mutation']])
        
        geometry = None
        dept = parcel_id[:2] if len(parcel_id) >= 2 else None
        
        # Try to find real building footprint
        if dept and dept in cadastre_buildings:
            buildings_gdf = cadastre_buildings[dept]
            
            # Get transaction point
            tx_point = group.geometry.iloc[0]
            
            # Find buildings containing this point
            try:
                containing = buildings_gdf[buildings_gdf.geometry.contains(tx_point)]
                
                if len(containing) > 0:
                    geometry = containing.iloc[0].geometry
                    with_real_building += 1
                else:
                    # Find nearest building within ~30m
                    distances = buildings_gdf.geometry.distance(tx_point)
                    min_dist = distances.min()
                    
                    if min_dist < 0.0003:  # ~30m
                        nearest_idx = distances.idxmin()
                        geometry = buildings_gdf.loc[nearest_idx].geometry
                        with_real_building += 1
            except Exception:
                pass
        
        # Fallback: rectangular shape
        if geometry is None:
            try:
                if len(group) == 1:
                    point = group.geometry.iloc[0]
                    # Small rectangular building footprint
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
                
                with_rectangle += 1
                
            except Exception:
                skipped += 1
                continue
        
        if geometry is None or geometry.is_empty:
            skipped += 1
            continue
        
        aggregated_data.append({
            'level': 'parcel',
            'code': parcel_id,
            'name': f"Parcel {parcel_id}",
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=transactions_gdf.crs)
    
    print(f"\n{'='*60}")
    print(f"✓ Aggregated into {len(result_gdf):,} building plots")
    print(f"{'='*60}")
    
    if len(result_gdf) > 0:
        total = with_real_building + with_rectangle
        pct_real = with_real_building / total * 100 if total > 0 else 0
        print(f"  With REAL building footprints: {with_real_building:,} ({pct_real:.1f}%)")
        print(f"  With rectangular shapes: {with_rectangle:,} ({100-pct_real:.1f}%)")
        print(f"  Skipped: {skipped:,}")
        print(f"  Median price range: €{result_gdf['median_price'].min():.0f} - €{result_gdf['median_price'].max():.0f}/m²")
    
    return result_gdf


def main():
    print("\n" + "="*60)
    print("REGENERATING PARCEL AGGREGATIONS (TEST MODE)")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis test version uses:")
    print("  - 50% sampled cadastre buildings (real footprints)")
    print("  - Rectangular shapes for remaining parcels")
    print("  - Max 200,000 parcels")
    
    overall_start = time.time()
    
    # Step 1: Delete existing parcel aggregates ONLY
    print("\n" + "="*60)
    print("STEP 1: DELETING EXISTING PARCEL AGGREGATES")
    print("="*60)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM price_aggregates WHERE level = 'parcel'"))
        count_before = result.scalar()
        print(f"  Existing parcel records: {count_before:,}")
        
        conn.execute(text("DELETE FROM price_aggregates WHERE level = 'parcel'"))
        conn.commit()
        print("✓ Deleted parcel aggregates")
    
    # Step 2: Load transactions
    gdf = load_transactions_from_db()
    
    # Step 3: Regenerate parcels
    print("\n" + "="*60)
    print("STEP 2: REGENERATING PARCEL AGGREGATES")
    print("="*60)
    
    start = time.time()
    parcel_agg = aggregate_by_parcel_test(gdf, max_parcels=200000, sample_pct=0.5)
    parcel_time = time.time() - start
    print(f"\n  Processing time: {parcel_time/60:.1f} minutes")
    
    # Step 4: Save to database
    print("\n" + "="*60)
    print("STEP 3: SAVING TO DATABASE")
    print("="*60)
    
    save_aggregates_to_postgres(parcel_agg, engine, if_exists='append')
    
    # Verify
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT level, COUNT(*) as count, AVG(median_price) as avg_price
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
        print("\nAggregation Summary:")
        print("-" * 50)
        for row in result:
            print(f"  {row[0]:12} | {row[1]:>8,} zones | €{row[2]:>6,.0f}/m²")
        print("-" * 50)
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("✓ PARCEL REGENERATION COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Parcels created: {len(parcel_agg):,}")
    print("\nRestart webapp to see changes: docker-compose restart webapp")
    print("="*60)


if __name__ == "__main__":
    main()
