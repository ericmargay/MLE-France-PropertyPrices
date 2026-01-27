"""
Spatial aggregation functions for property price analysis
WITH CONFIDENCE SCORING based on volume, volatility, and freshness

PRICE ESTIMATION METHODOLOGY
============================
This module estimates current property prices from historical transaction data
using the following approach:

1. TIME-WEIGHTED AVERAGE (Recency Weighting)
   - Recent transactions are weighted more heavily than older ones
   - Uses exponential decay: weight = exp(-days_since_transaction / tau)
   - Tau calibrated so half_life (180 days) gives 50% weight
   - Weights calculated relative to LAST transaction date in dataset,
     not the current date, to properly weight within the data period

2. INFLATION ADJUSTMENT (Market Trend Correction)
   - Historical prices are adjusted to present-day values
   - Uses French real estate price index (based on INSEE Notaires indices)
   - Annual adjustment rate: ~1.5% national average (configurable)
   - Applied per-transaction before aggregation

3. CONFIDENCE SCORING (Estimate Reliability: 0-100)
   Three components:
   
   a) VOLUME SCORE (0-40 points)
      - More transactions = more reliable estimate
      - >=100: 40 | >=50: 35 | >=30: 30 | >=20: 25 | >=10: 15 | >=5: 8 | <5: 5
   
   b) VOLATILITY SCORE (0-40 points)  
      - Lower price variance = more consistent market = higher confidence
      - Based on Coefficient of Variation (CV = std_dev / median)
      - CV <10%: 40 | <15%: 35 | <20%: 30 | <25%: 25 | <35%: 18 | <50%: 10 | >=50%: 5
   
   c) DATA RECENCY SCORE (0-20 points)
      - More recent data = more relevant to current market
      - Calculated relative to data period end (not current date)
      - <=30d: 20 | <=60d: 18 | <=90d: 15 | <=120d: 12 | <=180d: 10 | <=270d: 6 | <=365d: 4 | >365d: 2

4. CONFIDENCE INTERVAL ADJUSTMENT
   - Base interval: Interquartile Range (IQR = Q75 - Q25)
   - Wider intervals for lower confidence estimates
   - Multipliers: Very High (1.0x) | High (1.15x) | Medium (1.35x) | Low (1.7x) | Very Low (2.2x)

5. OUTPUT METRICS
   - median_price: Inflation-adjusted median price per m²
   - weighted_price: Time-weighted and inflation-adjusted average
   - lower_bound / upper_bound: Confidence-adjusted price range
   - confidence_score: 0-100 reliability score
   - estimate_quality: Human-readable quality label

REFERENCES
----------
- INSEE Notaires price indices: https://www.insee.fr/fr/statistiques/serie/001769365
- French real estate market reports: https://www.notaires.fr/fr/immobilier-fiscalite
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


# =============================================================================
# CONFIGURATION: Market Adjustment Parameters
# =============================================================================

# Data reference date (end of transaction data period)
DATA_REFERENCE_DATE = datetime(2023, 12, 31)

# Current estimation date (target date for price prediction)
ESTIMATION_DATE = datetime(2025, 1, 1)

# Annual price appreciation rate for French real estate
# Based on INSEE Notaires indices - can be updated with actual data
# French market 2023-2024 was relatively flat, slight recovery 2024-2025
ANNUAL_APPRECIATION_RATE = 0.015  # 1.5% average annual appreciation

# Time-weighting half-life in days
# A transaction from 180 days before the last transaction has 50% weight
DECAY_HALF_LIFE_DAYS = 180


def format_postcode(x):
    """
    Convert postcode to 5-digit string with leading zeros.
    
    French postcodes are always 5 digits (e.g., '01000', '75001').
    CSV import may read them as floats, losing leading zeros.
    
    Examples:
        1630.0  -> '01630'
        75001   -> '75001'
        '1234'  -> '01234'
    """
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


def get_inflation_adjustment_factor(transaction_date):
    """
    Calculate inflation adjustment factor to bring historical price to present value.
    
    Methodology:
    ------------
    Uses compound growth formula: factor = (1 + annual_rate) ^ years_elapsed
    
    Parameters:
    -----------
    transaction_date : datetime or str
        Date of the original transaction
    
    Returns:
    --------
    float : Multiplicative factor to adjust price to present value
    
    Example:
    --------
    A transaction from Jan 2023 with 1.5% annual rate:
    - Years elapsed to Jan 2025: 2.0 years
    - Factor: (1.015)^2 = 1.0302
    - €5000/m² in 2023 -> €5151/m² in 2025
    """
    if pd.isna(transaction_date):
        return 1.0
    
    # Convert to datetime if needed
    if isinstance(transaction_date, str):
        transaction_date = pd.to_datetime(transaction_date)
    
    # Calculate years elapsed
    days_elapsed = (ESTIMATION_DATE - transaction_date).days
    years_elapsed = days_elapsed / 365.25
    
    # Only forward adjustment (don't deflate)
    if years_elapsed < 0:
        years_elapsed = 0
    
    # Compound growth formula
    adjustment_factor = (1 + ANNUAL_APPRECIATION_RATE) ** years_elapsed
    
    return adjustment_factor


def calculate_weighted_price(group, apply_inflation=True):
    """
    Calculate time-weighted average price with optional inflation adjustment.
    
    Methodology:
    ------------
    1. INFLATION ADJUSTMENT (if enabled):
       Each transaction price is adjusted to present value using the 
       compound growth formula based on time elapsed since transaction.
    
    2. TIME-DECAY WEIGHTING:
       More recent transactions receive higher weights using exponential decay:
       
           weight_i = exp(-days_old_i / tau)
       
       where tau is calibrated so that a transaction from DECAY_HALF_LIFE_DAYS
       ago has 50% weight:  tau = DECAY_HALF_LIFE_DAYS / ln(2)
       
       "Days old" is calculated relative to the MOST RECENT transaction in the
       group, not the current date. This ensures proper relative weighting
       within the data period.
    
    3. WEIGHTED AVERAGE:
       Final price = sum(price_i * weight_i) / sum(weight_i)
    
    Parameters:
    -----------
    group : pd.DataFrame
        Transaction data with 'price_per_m2' and 'date_mutation' columns
    apply_inflation : bool
        Whether to adjust prices for inflation (default: True)
    
    Returns:
    --------
    float : Weighted average price per m² (inflation-adjusted if enabled)
    
    Statistical Note:
    -----------------
    The exponential decay function was chosen because:
    - It's widely used in time-series weighting (e.g., EMA in finance)
    - It has a clear interpretation (half-life parameter)
    - It smoothly decreases without sudden cutoffs
    - It's computationally efficient
    """
    if len(group) == 0:
        return np.nan
    
    group = group.copy()
    
    # Step 1: Apply inflation adjustment to each transaction
    if apply_inflation:
        group['adjusted_price'] = group.apply(
            lambda row: row['price_per_m2'] * get_inflation_adjustment_factor(row['date_mutation']),
            axis=1
        )
        price_col = 'adjusted_price'
    else:
        price_col = 'price_per_m2'
    
    # Step 2: Calculate days old relative to most recent transaction
    # This ensures proper relative weighting within the data period
    most_recent = pd.to_datetime(group['date_mutation']).max()
    group['days_old'] = (most_recent - pd.to_datetime(group['date_mutation'])).dt.days
    
    # Step 3: Calculate exponential decay weights
    # tau = half_life / ln(2), so that weight = 0.5 when days_old = half_life
    tau = DECAY_HALF_LIFE_DAYS / np.log(2)
    weights = np.exp(-group['days_old'] / tau)
    
    # Step 4: Calculate weighted average
    weighted_price = np.average(group[price_col], weights=weights)
    
    return weighted_price


def calculate_price_statistics(group):
    """
    Calculate comprehensive price statistics with confidence scoring.
    
    This is the main price estimation function that produces all metrics
    needed for the visualization and analysis.
    
    Methodology:
    ------------
    See module docstring for full methodology description.
    
    Parameters:
    -----------
    group : pd.DataFrame
        Transaction data with 'price_per_m2' and 'date_mutation' columns
    
    Returns:
    --------
    dict : Complete set of price metrics:
        - median_price: Inflation-adjusted median (EUR/m²)
        - weighted_price: Time-weighted, inflation-adjusted average (EUR/m²)
        - std_dev: Standard deviation of adjusted prices
        - transaction_count: Number of transactions
        - lower_bound: Lower confidence bound (EUR/m²)
        - upper_bound: Upper confidence bound (EUR/m²)
        - last_transaction_date: Date of most recent transaction
        - confidence_score: 0-100 reliability score
        - estimate_quality: 'Very High', 'High', 'Medium', 'Low', 'Very Low'
    
    Confidence Score Interpretation:
    --------------------------------
    85-100: Very High - Large sample, low variance, recent data
    70-84:  High      - Good reliability for most decisions
    55-69:  Medium    - Usable estimate with some uncertainty  
    35-54:  Low       - Use with caution, consider nearby areas
    0-34:   Very Low  - Indicative only, high uncertainty
    """
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
    
    group = group.copy()
    
    # =================================================================
    # STEP 1: Adjust all prices for inflation
    # =================================================================
    group['adjusted_price'] = group.apply(
        lambda row: row['price_per_m2'] * get_inflation_adjustment_factor(row['date_mutation']),
        axis=1
    )
    
    # =================================================================
    # STEP 2: Calculate basic statistics on adjusted prices
    # =================================================================
    median_price = group['adjusted_price'].median()
    std_dev = group['adjusted_price'].std()
    
    # Time-weighted average
    weighted_price = calculate_weighted_price(group, apply_inflation=True)
    
    # Quartiles for confidence interval
    q25 = group['adjusted_price'].quantile(0.25)
    q75 = group['adjusted_price'].quantile(0.75)
    
    # Transaction metadata
    transaction_count = len(group)
    last_date = pd.to_datetime(group['date_mutation']).max()
    
    # =================================================================
    # STEP 3: Calculate confidence score components
    # =================================================================
    
    # Days since last transaction (relative to data reference date)
    # This gives a fair freshness score within the data period
    days_since_last = (DATA_REFERENCE_DATE - last_date).days if pd.notna(last_date) else 999
    days_since_last = max(0, days_since_last)
    
    # Coefficient of variation (normalized volatility measure)
    cv = (std_dev / median_price) if median_price > 0 and not np.isnan(std_dev) else 1.0
    
    # --- VOLUME SCORE (0-40 points) ---
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
    
    # --- VOLATILITY SCORE (0-40 points) ---
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
    
    # --- FRESHNESS SCORE (0-20 points) ---
    if days_since_last <= 30:
        freshness_score = 20
    elif days_since_last <= 60:
        freshness_score = 18
    elif days_since_last <= 90:
        freshness_score = 15
    elif days_since_last <= 120:
        freshness_score = 12
    elif days_since_last <= 180:
        freshness_score = 10
    elif days_since_last <= 270:
        freshness_score = 6
    elif days_since_last <= 365:
        freshness_score = 4
    else:
        freshness_score = 2
    
    # --- TOTAL CONFIDENCE SCORE ---
    confidence_score = int(volume_score + volatility_score + freshness_score)
    
    # --- QUALITY LABEL ---
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
    
    # =================================================================
    # STEP 4: Calculate confidence-adjusted price bounds
    # =================================================================
    
    # Base interval: Interquartile Range
    iqr = q75 - q25 if not np.isnan(q75 - q25) else median_price * 0.2
    
    # Widen interval for lower confidence estimates
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
    
    adjusted_width = iqr * multiplier
    
    # Ensure minimum width (at least +/-5% of median)
    min_width = median_price * 0.10
    if adjusted_width < min_width:
        adjusted_width = min_width
    
    # Calculate final bounds
    lower_bound = median_price - adjusted_width / 2
    upper_bound = median_price + adjusted_width / 2
    
    # Ensure lower bound is reasonable
    lower_bound = max(100, lower_bound)
    
    return {
        'median_price': float(median_price),
        'weighted_price': float(weighted_price),
        'std_dev': float(std_dev) if not np.isnan(std_dev) else 0.0,
        'transaction_count': int(transaction_count),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
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
    
    sources = [
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/contours-codes-postaux/exports/geojson?lang=fr&timezone=Europe%2FParis",
        "https://www.data.gouv.fr/fr/datasets/r/eb36371a-761d-44a8-93ec-3d728bec17ce",
    ]
    
    for url in sources:
        try:
            print(f"  Downloading postcode boundaries...")
            response = requests.get(url, timeout=120)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                
                if 'zip' in content_type or url.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for name in z.namelist():
                            if name.endswith('.shp') or name.endswith('.geojson'):
                                z.extractall(cache_dir)
                                break
                    for f in os.listdir(cache_dir):
                        if f.endswith('.shp'):
                            gdf = gpd.read_file(os.path.join(cache_dir, f))
                            break
                        elif f.endswith('.geojson'):
                            gdf = gpd.read_file(os.path.join(cache_dir, f))
                            break
                else:
                    gdf = gpd.read_file(io.BytesIO(response.content))
                
                gdf.columns = gdf.columns.str.lower()
                
                postcode_col = None
                for col in ['code_postal', 'postal_code', 'postcode', 'cp', 'code']:
                    if col in gdf.columns:
                        postcode_col = col
                        break
                
                if postcode_col and postcode_col != 'code_postal':
                    gdf = gdf.rename(columns={postcode_col: 'code_postal'})
                
                if 'code_postal' in gdf.columns:
                    gdf['code_postal'] = gdf['code_postal'].astype(str).str.zfill(5)
                
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
    Falls back to convex hull from transaction points if boundary not available.
    """
    
    print(f"\n{'='*60}")
    print(f"Aggregating by POSTCODE (Official Boundaries)")
    print(f"{'='*60}")
    
    print("Loading official postcode boundaries...")
    postcode_boundaries = download_postcode_boundaries()
    
    use_official_boundaries = postcode_boundaries is not None and len(postcode_boundaries) > 0
    
    if use_official_boundaries:
        print(f"  ✓ Using {len(postcode_boundaries):,} official postcode polygons")
        postcode_geom_lookup = {}
        for _, row in postcode_boundaries.iterrows():
            pc = format_postcode(row.get('code_postal', ''))
            if pc:
                postcode_geom_lookup[pc] = row.geometry
        print(f"  ✓ Indexed {len(postcode_geom_lookup):,} postcodes")
    else:
        print("  ⚠️  Official boundaries not available, using fallback approach")
        postcode_geom_lookup = {}
    
    if 'code' not in communes_gdf.columns:
        if 'insee' in communes_gdf.columns:
            communes_gdf['code'] = communes_gdf['insee']
        else:
            communes_gdf['code'] = range(len(communes_gdf))
    
    print("Formatting postcodes...")
    transactions_gdf = transactions_gdf.copy()
    transactions_gdf['code_postal'] = transactions_gdf['code_postal'].apply(format_postcode)
    
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
        
        stats = calculate_price_statistics(postcode_transactions[['price_per_m2', 'date_mutation']])
        
        geometry = None
        
        if postcode in postcode_geom_lookup:
            geometry = postcode_geom_lookup[postcode]
            with_official_boundary += 1
        else:
            try:
                points = postcode_transactions.geometry.tolist()
                if len(points) >= 3:
                    multi_point = MultiPoint(points)
                    geometry = multi_point.convex_hull
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
    Handles large files and timeouts gracefully.
    """
    import gzip
    import json
    
    os.makedirs(cache_dir, exist_ok=True)
    
    dept = str(departement_code).zfill(2) if str(departement_code).isdigit() else str(departement_code)
    
    cache_file = os.path.join(cache_dir, f'batiments_{dept}.geojson')
    failed_marker = os.path.join(cache_dir, f'batiments_{dept}.failed')
    
    if os.path.exists(failed_marker):
        return None
    
    if os.path.exists(cache_file):
        try:
            return gpd.read_file(cache_file)
        except Exception:
            pass
    
    large_depts = ['59', '75', '69', '13', '31', '33', '34', '06', '83', '92', '93', '94', '77', '78', '91', '95']
    if dept in large_depts:
        print(f"    Skipping large dept {dept} (will use estimated footprints)")
        with open(failed_marker, 'w') as f:
            f.write('skipped - large file')
        return None
    
    url = f"https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/geojson/departements/{dept}/cadastre-{dept}-batiments.json.gz"
    
    try:
        print(f"    Downloading building footprints for dept {dept}...")
        
        response = requests.get(url, timeout=60, stream=True)
        
        if response.status_code != 200:
            print(f"    Warning: HTTP {response.status_code} for dept {dept}")
            return None
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 100 * 1024 * 1024:
            print(f"    Skipping dept {dept} - file too large")
            with open(failed_marker, 'w') as f:
                f.write('skipped - too large')
            return None
        
        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024):
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded > 100 * 1024 * 1024:
                print(f"    Stopping download for dept {dept} - exceeds 100MB")
                with open(failed_marker, 'w') as f:
                    f.write('stopped - exceeded size limit')
                return None
        
        content = b''.join(chunks)
        
        try:
            decompressed = gzip.decompress(content)
        except Exception as e:
            print(f"    Warning: Decompression failed for dept {dept}: {e}")
            return None
        
        try:
            geojson_data = json.loads(decompressed)
        except Exception as e:
            print(f"    Warning: JSON parse failed for dept {dept}: {e}")
            return None
        
        if 'features' not in geojson_data or len(geojson_data['features']) == 0:
            print(f"    Warning: No features in dept {dept}")
            return None
        
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs="EPSG:4326")
        
        try:
            gdf.to_file(cache_file, driver='GeoJSON')
        except Exception:
            pass
        
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
    
    parcels_df['dept_code'] = parcels_df['id_parcelle'].str[:2]
    
    departments = parcels_df['dept_code'].unique()
    departments = [d for d in departments if d and len(d) >= 2]
    print(f"  Departments with parcels: {len(departments)}")
    
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
        
        if dept and dept in cadastre_buildings:
            buildings_gdf = cadastre_buildings[dept]
            
            tx_points = group.geometry.tolist()
            
            for tx_point in tx_points:
                containing = buildings_gdf[buildings_gdf.contains(tx_point)]
                
                if len(containing) > 0:
                    geometry = containing.iloc[0].geometry
                    with_building_footprint += 1
                    break
                else:
                    distances = buildings_gdf.geometry.distance(tx_point)
                    nearest_idx = distances.idxmin()
                    nearest_dist = distances[nearest_idx]
                    
                    if nearest_dist < 0.0005:
                        geometry = buildings_gdf.loc[nearest_idx].geometry
                        with_building_footprint += 1
                        break
        
        if geometry is None:
            try:
                if len(group) == 1:
                    point = group.geometry.iloc[0]
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
