"""
Spatial aggregation functions for property price analysis
WITH CONFIDENCE SCORING based on volume, volatility, and freshness

IMPROVEMENTS:
1. Postcodes: Uses transaction clustering for more granular divisions
2. Parcels: Downloads and uses actual French cadastre building footprints
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry, WKTElement
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union, voronoi_diagram
from shapely import affinity
import os
import requests
import warnings
warnings.filterwarnings('ignore')


def format_postcode(x):
    """Convert postcode to 5-digit string with leading zeros."""
    if pd.isna(x) or str(x).strip() == '':
        return ''
    try:
        num = int(float(x))
        return str(num).zfill(5)
    except (ValueError, TypeError):
        cleaned = str(x).strip()
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        return cleaned.zfill(5)


def calculate_weighted_price(group, recency_weight=True, decay_days=365):
    """Calculate time-weighted average price"""
    if len(group) == 0:
        return np.nan
    
    if not recency_weight:
        return group['price_per_m2'].mean()
    
    now = datetime.now()
    group = group.copy()
    group['days_old'] = (now - pd.to_datetime(group['date_mutation'])).dt.days
    weights = np.exp(-group['days_old'] / decay_days)
    weighted_price = np.average(group['price_per_m2'], weights=weights)
    
    return weighted_price


def calculate_price_statistics(group):
    """Calculate comprehensive price statistics with confidence scoring"""
    if len(group) == 0:
        return {
            'median_price': np.nan,
            'weighted_price': np.nan,
            'std_dev': np.nan,
            'transaction_count': 0,
            'lower_bound': np.nan,
            'upper_bound': np.nan,
            'last_transaction_date': None,
            'confidence_score': 0,
            'estimate_quality': 'No Data'
        }
    
    median_price = group['price_per_m2'].median()
    std_dev = group['price_per_m2'].std()
    weighted_price = calculate_weighted_price(group, recency_weight=True)
    
    q25 = group['price_per_m2'].quantile(0.25)
    q75 = group['price_per_m2'].quantile(0.75)
    
    transaction_count = len(group)
    last_date = group['date_mutation'].max()
    
    days_since_last = (datetime.now() - pd.to_datetime(last_date)).days if pd.notna(last_date) else 999
    cv = (std_dev / median_price) if median_price > 0 and not np.isnan(std_dev) else 1.0
    
    # Volume score (0-40)
    if transaction_count >= 100:
        volume_score = 40
    elif transaction_count >= 50:
        volume_score = 35
    elif transaction_count >= 30:
        volume_score = 30
    elif transaction_count >= 20:
        volume_score = 25
    elif transaction_count >= 10:
        volume_score = 15
    elif transaction_count >= 5:
        volume_score = 8
    else:
        volume_score = 5
    
    # Volatility score (0-40)
    if cv < 0.10:
        volatility_score = 40
    elif cv < 0.15:
        volatility_score = 35
    elif cv < 0.20:
        volatility_score = 30
    elif cv < 0.25:
        volatility_score = 25
    elif cv < 0.35:
        volatility_score = 18
    elif cv < 0.50:
        volatility_score = 10
    else:
        volatility_score = 5
    
    # Freshness score (0-20)
    if days_since_last <= 30:
        freshness_score = 20
    elif days_since_last <= 60:
        freshness_score = 18
    elif days_since_last <= 90:
        freshness_score = 15
    elif days_since_last <= 120:
        freshness_score = 12
    elif days_since_last <= 180:
        freshness_score = 8
    elif days_since_last <= 270:
        freshness_score = 5
    else:
        freshness_score = 2
    
    confidence_score = int(volume_score + volatility_score + freshness_score)
    
    if confidence_score >= 85:
        quality = 'Very High'
    elif confidence_score >= 70:
        quality = 'High'
    elif confidence_score >= 55:
        quality = 'Medium'
    elif confidence_score >= 35:
        quality = 'Low'
    else:
        quality = 'Very Low'
    
    interval_width = q75 - q25 if not np.isnan(q75 - q25) else 0
    median_est = median_price
    
    if confidence_score >= 85:
        multiplier = 1.0
    elif confidence_score >= 70:
        multiplier = 1.15
    elif confidence_score >= 55:
        multiplier = 1.35
    elif confidence_score >= 35:
        multiplier = 1.7
    else:
        multiplier = 2.2
    
    adjusted_width = interval_width * multiplier
    min_width = median_est * 0.05
    if adjusted_width < min_width:
        adjusted_width = min_width
    
    q25_adjusted = median_est - adjusted_width / 2
    q75_adjusted = median_est + adjusted_width / 2
    q25_adjusted = max(100, q25_adjusted)
    
    return {
        'median_price': float(median_price),
        'weighted_price': float(weighted_price),
        'std_dev': float(std_dev) if not np.isnan(std_dev) else 0.0,
        'transaction_count': int(transaction_count),
        'lower_bound': float(q25_adjusted),
        'upper_bound': float(q75_adjusted),
        'last_transaction_date': last_date,
        'confidence_score': int(confidence_score),
        'estimate_quality': quality
    }


def spatial_join_and_aggregate(transactions_gdf, boundaries_gdf, 
                                 level_name, join_column, name_column=None):
    """Perform spatial join and aggregate prices by geographic boundaries"""
    print(f"\n{'='*60}")
    print(f"Aggregating by {level_name.upper()}")
    print(f"{'='*60}")
    
    print(f"Transactions: {len(transactions_gdf):,}")
    print(f"Boundaries: {len(boundaries_gdf):,}")
    
    if transactions_gdf.crs != boundaries_gdf.crs:
        transactions_gdf = transactions_gdf.to_crs(boundaries_gdf.crs)
    
    print("Performing spatial join...")
    joined = gpd.sjoin(transactions_gdf, boundaries_gdf, how='inner', predicate='within')
    
    print(f"Matched transactions: {len(joined):,}")
    
    aggregated_data = []
    
    for code in joined[join_column].unique():
        group = joined[joined[join_column] == code]
        stats = calculate_price_statistics(group[['price_per_m2', 'date_mutation']])
        
        if name_column and name_column in group.columns:
            name = group[name_column].iloc[0]
        else:
            name = code
        
        geometry = boundaries_gdf[boundaries_gdf[join_column] == code].geometry.iloc[0]
        
        aggregated_data.append({
            'level': level_name,
            'code': code,
            'name': name,
            'geometry': geometry,
            **stats
        })
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=boundaries_gdf.crs)
    
    print(f"✓ Aggregated into {len(result_gdf)} regions")
    if len(result_gdf) > 0:
        print(f"  Median price range: €{result_gdf['median_price'].min():.0f} - €{result_gdf['median_price'].max():.0f}/m²")
    
    return result_gdf


def download_postcode_boundaries(cache_dir='/app/data/geometries'):
    """
    Download official French postcode boundaries from data.gouv.fr
    Source: https://www.data.gouv.fr/fr/datasets/fond-de-carte-des-codes-postaux/
    """
    import zipfile
    import io
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'codes_postaux.geojson')
    
    if os.path.exists(cache_file):
        try:
            print("  Loading cached postcode boundaries...")
            return gpd.read_file(cache_file)
        except Exception:
            pass
    
    # Try multiple sources for French postcode boundaries
    sources = [
        # OpenDataSoft - most reliable
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/contours-codes-postaux/exports/geojson?lang=fr&timezone=Europe%2FParis",
        # Alternative from data.gouv.fr
        "https://www.data.gouv.fr/fr/datasets/r/eb36371a-761d-44a8-93ec-3d728bec17ce",
    ]
    
    for url in sources:
        try:
            print(f"  Downloading postcode boundaries...")
            response = requests.get(url, timeout=120)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                
                if 'zip' in content_type or url.endswith('.zip'):
                    # Handle ZIP file
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for name in z.namelist():
                            if name.endswith('.shp') or name.endswith('.geojson'):
                                z.extractall(cache_dir)
                                break
                    # Find the extracted file
                    for f in os.listdir(cache_dir):
                        if f.endswith('.shp'):
                            gdf = gpd.read_file(os.path.join(cache_dir, f))
                            break
                        elif f.endswith('.geojson'):
                            gdf = gpd.read_file(os.path.join(cache_dir, f))
                            break
                else:
                    # Direct GeoJSON
                    gdf = gpd.read_file(io.BytesIO(response.content))
                
                # Standardize column names
                gdf.columns = gdf.columns.str.lower()
                
                # Find postcode column
                postcode_col = None
                for col in ['code_postal', 'postal_code', 'postcode', 'cp', 'code']:
                    if col in gdf.columns:
                        postcode_col = col
                        break
                
                if postcode_col and postcode_col != 'code_postal':
                    gdf = gdf.rename(columns={postcode_col: 'code_postal'})
                
                # Ensure postcode is string with leading zeros
                if 'code_postal' in gdf.columns:
                    gdf['code_postal'] = gdf['code_postal'].astype(str).str.zfill(5)
                
                # Save to cache
                gdf.to_file(cache_file, driver='GeoJSON')
                
                print(f"  ✓ Downloaded {len(gdf):,} postcode boundaries")
                return gdf
                
        except Exception as e:
            print(f"  Warning: Failed to download from source: {e}")
            continue
    
    print("  Warning: Could not download postcode boundaries")
    return None


def aggregate_by_postcode(transactions_gdf, communes_gdf):
    """
    Aggregate by postcode using OFFICIAL POSTCODE BOUNDARIES.
    
    Downloads real French postcode polygons and uses them for visualization.
    Falls back to commune boundaries if postcode boundary not available.
    """
    
    print(f"\n{'='*60}")
    print(f"Aggregating by POSTCODE (Official Boundaries)")
    print(f"{'='*60}")
    
    # Download official postcode boundaries
    print("Loading official postcode boundaries...")
    postcode_boundaries = download_postcode_boundaries()
    
    use_official_boundaries = postcode_boundaries is not None and len(postcode_boundaries) > 0
    
    if use_official_boundaries:
        print(f"  ✓ Using {len(postcode_boundaries):,} official postcode polygons")
        # Create lookup dictionary for fast access
        postcode_geom_lookup = {}
        for _, row in postcode_boundaries.iterrows():
            pc = format_postcode(row.get('code_postal', ''))
            if pc:
                postcode_geom_lookup[pc] = row.geometry
        print(f"  ✓ Indexed {len(postcode_geom_lookup):,} postcodes")
    else:
        print("  ⚠️  Official boundaries not available, using commune-based approach")
        postcode_geom_lookup = {}
    
    # Ensure communes have proper code column
    if 'code' not in communes_gdf.columns:
        if 'insee' in communes_gdf.columns:
            communes_gdf['code'] = communes_gdf['insee']
        else:
            communes_gdf['code'] = range(len(communes_gdf))
    
    # Format postcodes
    print("Formatting postcodes...")
    transactions_gdf = transactions_gdf.copy()
    transactions_gdf['code_postal'] = transactions_gdf['code_postal'].apply(format_postcode)
    
    # Group transactions by postcode
    print("Grouping transactions by postcode...")
    postcode_groups = transactions_gdf.groupby('code_postal')
    
    unique_postcodes = [pc for pc in postcode_groups.groups.keys() if pc and pc != '']
    print(f"Processing {len(unique_postcodes):,} unique postcodes...")
    
    aggregated_data = []
    with_official_boundary = 0
    with_fallback = 0
    
    for i, postcode in enumerate(unique_postcodes):
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1:,}/{len(unique_postcodes):,} postcodes processed")
        
        postcode_transactions = postcode_groups.get_group(postcode)
        
        if len(postcode_transactions) == 0:
            continue
        
        # Calculate statistics
        stats = calculate_price_statistics(postcode_transactions[['price_per_m2', 'date_mutation']])
        
        # Get geometry - prefer official boundary
        geometry = None
        
        if postcode in postcode_geom_lookup:
            geometry = postcode_geom_lookup[postcode]
            with_official_boundary += 1
        else:
            # Fallback: create convex hull from transaction points
            # or use union of communes where transactions occur
            try:
                points = postcode_transactions.geometry.tolist()
                if len(points) >= 3:
                    multi_point = MultiPoint(points)
                    geometry = multi_point.convex_hull
                    # Add small buffer to make it visible
                    if isinstance(geometry, (Point, LineString)):
                        geometry = geometry.buffer(0.002)
                    else:
                        geometry = geometry.buffer(0.0005)
                elif len(points) == 2:
                    geometry = LineString(points).buffer(0.002)
                else:
                    geometry = points[0].buffer(0.003)
                
                with_fallback += 1
            except Exception:
                continue
        
        if geometry is None or geometry.is_empty:
            continue
        
        # Ensure valid polygon
        if isinstance(geometry, MultiPolygon):
            geometry = max(geometry.geoms, key=lambda g: g.area)
        elif not isinstance(geometry, Polygon):
            try:
                geometry = geometry.convex_hull
                if not isinstance(geometry, Polygon):
                    geometry = geometry.buffer(0.001)
            except:
                continue
        
        aggregated_data.append({
            'level': 'postcode',
            'code': postcode,
            'name': f"{postcode}",
            'geometry': geometry,
            **stats
        })
    
    print(f"  ... {len(unique_postcodes):,}/{len(unique_postcodes):,} postcodes processed")
    
    result_gdf = gpd.GeoDataFrame(aggregated_data, crs=transactions_gdf.crs)
    
    print(f"\n✓ Created {len(result_gdf):,} postcode zones")
    print(f"  With official boundaries: {with_official_boundary:,}")
    print(f"  With generated boundaries: {with_fallback:,}")
    
    if len(result_gdf) > 0:
        print(f"  Median price range: €{result_gdf['median_price'].min():.0f} - €{result_gdf['median_price'].max():.0f}/m²")
    
    return result_gdf


def download_cadastre_batiments(departement_code, cache_dir='/app/data/cadastre'):
    """
    Download cadastre BATIMENTS (building footprints) for a department.
    This gives actual building shapes, not just parcel boundaries.
    
    Handles large files and timeouts gracefully.
    """
    import gzip
    import json
    
    os.makedirs(cache_dir, exist_ok=True)
    
    dept = str(departement_code).zfill(2) if str(departement_code).isdigit() else str(departement_code)
    
    cache_file = os.path.join(cache_dir, f'batiments_{dept}.geojson')
    failed_marker = os.path.join(cache_dir, f'batiments_{dept}.failed')
    
    # Skip if previously failed (avoid re-trying large files)
    if os.path.exists(failed_marker):
        return None
    
    # Use cache if available
    if os.path.exists(cache_file):
        try:
            return gpd.read_file(cache_file)
        except Exception:
            pass
    
    # Large departments - skip download, use fallback geometry
    # These files are 500MB+ and often timeout
    large_depts = ['59', '75', '69', '13', '31', '33', '34', '06', '83', '92', '93', '94', '77', '78', '91', '95']
    if dept in large_depts:
        print(f"    Skipping large dept {dept} (will use estimated footprints)")
        # Mark as failed to skip in future
        with open(failed_marker, 'w') as f:
            f.write('skipped - large file')
        return None
    
    url = f"https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/departements/{dept}/cadastre-{dept}-batiments.json.gz"
    
    try:
        print(f"    Downloading building footprints for dept {dept}...")
        
        # Stream download with timeout
        response = requests.get(url, timeout=60, stream=True)
        
        if response.status_code != 200:
            print(f"    Warning: HTTP {response.status_code} for dept {dept}")
            return None
        
        # Check content length - skip if too large (>100MB compressed)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 100 * 1024 * 1024:
            print(f"    Skipping dept {dept} - file too large ({int(content_length)/1024/1024:.0f}MB)")
            with open(failed_marker, 'w') as f:
                f.write('skipped - too large')
            return None
        
        # Download in chunks
        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded > 100 * 1024 * 1024:  # Stop if >100MB
                print(f"    Stopping download for dept {dept} - exceeds 100MB")
                with open(failed_marker, 'w') as f:
                    f.write('stopped - exceeded size limit')
                return None
        
        content = b''.join(chunks)
        
        # Decompress
        try:
            decompressed = gzip.decompress(content)
        except Exception as e:
            print(f"    Warning: Decompression failed for dept {dept}: {e}")
            return None
        
        # Parse JSON
        try:
            geojson_data = json.loads(decompressed)
        except Exception as e:
            print(f"    Warning: JSON parse failed for dept {dept}: {e}")
            return None
        
        # Convert to GeoDataFrame
        if 'features' not in geojson_data or len(geojson_data['features']) == 0:
            print(f"    Warning: No features in dept {dept}")
            return None
        
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs="EPSG:4326")
        
        # Save to cache
        try:
            gdf.to_file(cache_file, driver='GeoJSON')
        except Exception:
            pass  # Cache save failed, but we still have the data
        
        print(f"    ✓ Loaded {len(gdf):,} buildings for dept {dept}")
        return gdf
        
    except requests.exceptions.Timeout:
        print(f"    Warning: Timeout downloading dept {dept}")
        with open(failed_marker, 'w') as f:
            f.write('timeout')
        return None
    except requests.exceptions.RequestException as e:
        print(f"    Warning: Network error for dept {dept}: {e}")
        return None
    except Exception as e:
        print(f"    Warning: Error processing dept {dept}: {e}")
        return None


def aggregate_by_parcel(transactions_gdf, max_parcels=300000):
    """
    Aggregate property prices by cadastral parcel using ACTUAL BUILDING FOOTPRINTS.
    
    Downloads French cadastre building data (batiments) to get real building shapes
    instead of using rectangular buffers.
    """
    print(f"\n{'='*60}")
    print(f"Aggregating by BUILDING PLOT (Cadastre Building Footprints)")
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
    
    # Load cadastre building data for each department
    print(f"\nDownloading cadastre building footprints for {len(departments)} departments...")
    print("  (Large departments like Paris, Lyon, Marseille will use estimated footprints)")
    cadastre_buildings = {}
    
    for i, dept in enumerate(departments):
        if (i + 1) % 10 == 0:
            print(f"  ... processed {i + 1}/{len(departments)} departments ({len(cadastre_buildings)} loaded)")
        
        try:
            buildings_gdf = download_cadastre_batiments(dept)
            if buildings_gdf is not None and len(buildings_gdf) > 0:
                cadastre_buildings[dept] = buildings_gdf
        except Exception as e:
            print(f"    Warning: Failed to load dept {dept}: {e}")
            continue
    
    print(f"  ✓ Loaded building data for {len(cadastre_buildings)}/{len(departments)} departments")
    
    # Group by parcel
    print("\nGrouping by parcel ID...")
    parcel_groups = parcels_df.groupby('id_parcelle')
    
    total_parcels = len(parcel_groups)
    print(f"  Unique parcels: {total_parcels:,}")
    
    if total_parcels > max_parcels:
        print(f"  ⚠️  Limiting to top {max_parcels:,} parcels...")
        parcel_counts = parcel_groups.size().sort_values(ascending=False).head(max_parcels)
        selected_parcels = parcel_counts.index.tolist()
        parcels_df = parcels_df[parcels_df['id_parcelle'].isin(selected_parcels)]
        parcel_groups = parcels_df.groupby('id_parcelle')
    
    aggregated_data = []
    with_building_footprint = 0
    with_buffer = 0
    skipped = 0
    
    print(f"\nProcessing {len(parcel_groups):,} parcels...")
    
    for idx, (parcel_id, group) in enumerate(parcel_groups):
        if (idx + 1) % 50000 == 0:
            print(f"  ... {idx + 1:,}/{len(parcel_groups):,} parcels processed")
        
        if len(group) == 0:
            skipped += 1
            continue
        
        stats = calculate_price_statistics(group[['price_per_m2', 'date_mutation']])
        
        geometry = None
        dept = parcel_id[:2] if len(parcel_id) >= 2 else None
        
        # Try to find building footprint at transaction location
        if dept and dept in cadastre_buildings:
            buildings_gdf = cadastre_buildings[dept]
            
            # Get transaction point(s)
            tx_points = group.geometry.tolist()
            tx_centroid = group.geometry.unary_union.centroid
            
            # Find buildings that contain or are near the transaction point
            for tx_point in tx_points:
                # Check which buildings contain this point
                containing = buildings_gdf[buildings_gdf.contains(tx_point)]
                
                if len(containing) > 0:
                    geometry = containing.iloc[0].geometry
                    with_building_footprint += 1
                    break
                else:
                    # Find nearest building within 50m
                    distances = buildings_gdf.geometry.distance(tx_point)
                    nearest_idx = distances.idxmin()
                    nearest_dist = distances[nearest_idx]
                    
                    if nearest_dist < 0.0005:  # ~50m
                        geometry = buildings_gdf.loc[nearest_idx].geometry
                        with_building_footprint += 1
                        break
        
        # Fallback: create small building footprint from points
        if geometry is None:
            try:
                if len(group) == 1:
                    point = group.geometry.iloc[0]
                    # Create realistic building shape (~12m x 12m)
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
                
                with_buffer += 1
                
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
    
    print(f"\n✓ Aggregated into {len(result_gdf):,} building plots")
    if len(result_gdf) > 0:
        pct = with_building_footprint / len(result_gdf) * 100
        print(f"  With real building footprints: {with_building_footprint:,} ({pct:.1f}%)")
    print(f"  With estimated footprints: {with_buffer:,}")
    print(f"  Skipped: {skipped}")
    
    if len(result_gdf) > 0:
        print(f"  Median price range: €{result_gdf['median_price'].min():.0f} - €{result_gdf['median_price'].max():.0f}/m²")
    
    return result_gdf


def create_country_aggregate(transactions_gdf, country_code='FRA'):
    """Create a single country-level aggregate for Metropolitan France"""
    print(f"\n{'='*60}")
    print(f"Aggregating by COUNTRY")
    print(f"{'='*60}")
    
    metro_france = transactions_gdf[
        (transactions_gdf.geometry.x >= -5) & 
        (transactions_gdf.geometry.x <= 10) &
        (transactions_gdf.geometry.y >= 41) &
        (transactions_gdf.geometry.y <= 52)
    ]
    
    print(f"  Metropolitan France: {len(metro_france):,} transactions")
    
    stats = calculate_price_statistics(transactions_gdf[['price_per_m2', 'date_mutation']])
    
    try:
        regions_path = '/app/data/geometries/regions.geojson'
        if os.path.exists(regions_path):
            regions = gpd.read_file(regions_path)
            country_geometry = regions.geometry.unary_union
        else:
            country_geometry = metro_france.geometry.unary_union.convex_hull
    except:
        country_geometry = metro_france.geometry.unary_union.convex_hull
    
    result_gdf = gpd.GeoDataFrame([{
        'level': 'country',
        'code': country_code,
        'name': 'France',
        'geometry': country_geometry,
        **stats
    }], crs=transactions_gdf.crs)
    
    print(f"✓ Country aggregate created")
    return result_gdf


def save_aggregates_to_postgres(aggregates_gdf, engine, if_exists='append'):
    """Save aggregated data to PostgreSQL with PostGIS geometry"""
    print(f"\nSaving {len(aggregates_gdf)} aggregates to PostgreSQL...")
    
    gdf_copy = aggregates_gdf.copy()
    
    def to_multipolygon(geom):
        if geom is None:
            return None
        if isinstance(geom, MultiPolygon):
            return geom
        elif isinstance(geom, Polygon):
            return MultiPolygon([geom])
        elif isinstance(geom, Point):
            buffered = geom.buffer(0.001)
            return MultiPolygon([buffered]) if isinstance(buffered, Polygon) else None
        else:
            try:
                buffered = geom.buffer(0.001)
                return MultiPolygon([buffered]) if isinstance(buffered, Polygon) else None
            except:
                return None
    
    gdf_copy['geometry'] = gdf_copy['geometry'].apply(to_multipolygon)
    
    initial_count = len(gdf_copy)
    gdf_copy = gdf_copy[gdf_copy['geometry'].notna()]
    if len(gdf_copy) < initial_count:
        print(f"  Removed {initial_count - len(gdf_copy)} invalid geometries")
    
    gdf_copy['geom'] = gdf_copy['geometry'].apply(
        lambda x: WKTElement(x.wkt, srid=4326) if x is not None else None
    )
    gdf_copy = gdf_copy.drop(columns=['geometry'])
    
    columns = ['level', 'code', 'name', 'median_price', 'weighted_price', 
               'std_dev', 'transaction_count', 'lower_bound', 'upper_bound', 
               'last_transaction_date', 'confidence_score', 'estimate_quality', 'geom']
    
    gdf_copy[columns].to_sql(
        'price_aggregates',
        engine,
        if_exists=if_exists,
        index=False,
        dtype={'geom': Geometry('GEOMETRY', srid=4326)},
        method='multi',
        chunksize=1000
    )
    
    print(f"✓ Saved successfully!")
