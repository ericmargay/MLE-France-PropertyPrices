#!/usr/bin/env python3
"""
OPTIMIZED ML Price Aggregation with Full Cadastre Support
=========================================================

Uses all available CPUs for parcel processing.
Processes ALL parcels with real building footprints.

Usage:
    docker-compose run --rm etl python regenerate_with_ml_optimized.py

Environment variables:
    N_WORKERS: Number of parallel workers (default: CPU count)
    SKIP_PARCELS: Set to 'true' to skip parcel level
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely import wkt
from sqlalchemy import create_engine, text
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
GEOMETRIES_DIR = '/app/data/geometries'
CADASTRE_CACHE_DIR = '/app/data/cadastre'
MODEL_PATH = '/app/data/ml_price_model.joblib'

SKIP_PARCELS = os.getenv('SKIP_PARCELS', 'false').lower() == 'true'
N_WORKERS = int(os.getenv('N_WORKERS', cpu_count()))

# No limit on parcels - process ALL
MAX_PARCELS = None  # Set to int to limit, None for all

print(f"Configuration:")
print(f"  Workers: {N_WORKERS}")
print(f"  Skip parcels: {SKIP_PARCELS}")
print(f"  Max parcels: {MAX_PARCELS or 'ALL'}")

engine = create_engine(DATABASE_URL)


def download_boundaries():
    """Download French administrative boundaries if missing."""
    import requests
    
    os.makedirs(GEOMETRIES_DIR, exist_ok=True)
    
    # URLs for French administrative boundaries
    urls = {
        'regions': 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson',
        'departements': 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson',
        'communes': 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes.geojson'
    }
    
    downloaded = []
    
    for name, url in urls.items():
        filepath = os.path.join(GEOMETRIES_DIR, f'{name}.geojson')
        
        if os.path.exists(filepath):
            print(f"  ✓ {name}.geojson already exists")
            downloaded.append(name)
            continue
        
        print(f"  Downloading {name}.geojson...")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"  ✓ {name}.geojson downloaded ({len(response.text)//1024} KB)")
            downloaded.append(name)
            
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")
    
    return downloaded

# =============================================================================
# ML MODEL (with threading support)
# =============================================================================

class RobustPriceModel:
    """ML model with LightGBM using all threads."""
    
    def __init__(self):
        self.model = None
        self.temporal_trend = None
        self.prediction_date = datetime(2026, 1, 1)
        self.dept_medians = {}
        self.global_median = None
        self.global_std = None
        
    def fit(self, transactions_df):
        """Train the model using all available threads."""
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        
        print(f"\nTraining samples: {len(transactions_df):,}")
        
        df = transactions_df.copy()
        
        # Step 1: Hierarchical priors
        print("\nStep 1: Computing hierarchical priors...")
        self.global_median = df['price_per_m2'].median()
        self.global_std = df['price_per_m2'].std()
        print(f"  Global median: €{self.global_median:,.0f}/m²")
        
        for dept in df['code_departement'].unique():
            dept_data = df[df['code_departement'] == dept]['price_per_m2']
            if len(dept_data) >= 10:
                self.dept_medians[dept] = dept_data.median()
        print(f"  Computed medians for {len(self.dept_medians)} departments")
        
        # Step 2: Temporal trend
        print("\nStep 2: Estimating temporal trend...")
        df['date_mutation'] = pd.to_datetime(df['date_mutation'])
        df['year_frac'] = df['date_mutation'].dt.year + df['date_mutation'].dt.dayofyear / 365.25
        
        valid = df[(df['price_per_m2'] > 0) & (df['year_frac'] > 2020)]
        if len(valid) > 1000:
            from sklearn.linear_model import LinearRegression
            X_time = valid['year_frac'].values.reshape(-1, 1)
            y_log = np.log(valid['price_per_m2'].values)
            
            lr = LinearRegression()
            lr.fit(X_time, y_log)
            self.temporal_trend = lr.coef_[0]
            print(f"  Annual trend: {self.temporal_trend*100:.2f}%")
        else:
            self.temporal_trend = 0
            
        # Step 3: Features
        print("\nStep 3: Engineering features...")
        df['log_price'] = np.log(df['price_per_m2'])
        
        # Adjust to prediction date
        pred_year = self.prediction_date.year + self.prediction_date.timetuple().tm_yday / 365.25
        df['years_to_pred'] = pred_year - df['year_frac']
        df['log_price_adj'] = df['log_price'] + self.temporal_trend * df['years_to_pred']
        
        le_dept = LabelEncoder()
        df['dept_encoded'] = le_dept.fit_transform(df['code_departement'].astype(str))
        self.dept_encoder = le_dept
        
        le_type = LabelEncoder()
        df['type_encoded'] = le_type.fit_transform(df['type_local'].fillna('Unknown'))
        self.type_encoder = le_type
        
        df['log_surface'] = np.log1p(df['surface_reelle_bati'].fillna(50))
        df['lat_squared'] = df['latitude'] ** 2
        df['lon_squared'] = df['longitude'] ** 2
        df['lat_lon_interaction'] = df['latitude'] * df['longitude']
        
        features = ['latitude', 'longitude', 'dept_encoded', 'type_encoded', 
                   'log_surface', 'lat_squared', 'lon_squared', 'lat_lon_interaction']
        
        X = df[features].values
        y = df['log_price_adj'].values
        
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Step 4: Train LightGBM with all threads
        print(f"\nStep 4: Training LightGBM with {N_WORKERS} threads...")
        
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=12,
            learning_rate=0.05,
            num_leaves=64,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=N_WORKERS,  # Use all CPUs
            num_threads=N_WORKERS
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"  CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Final training
        self.model.fit(X, y)
        self.feature_names = features
        
        # Feature importance
        print("\n  Feature importance:")
        for name, imp in sorted(zip(features, self.model.feature_importances_), 
                                key=lambda x: -x[1])[:5]:
            print(f"    {name}: {imp:.0f}")
        
        return self
    
    def predict_zone(self, transactions_df, geometry=None, parent_estimate=None):
        """Predict price for a zone with confidence intervals."""
        if len(transactions_df) == 0:
            return None
            
        df = transactions_df.copy()
        
        # Prepare features
        try:
            df['dept_encoded'] = self.dept_encoder.transform(
                df['code_departement'].astype(str))
        except:
            df['dept_encoded'] = 0
            
        try:
            df['type_encoded'] = self.type_encoder.transform(
                df['type_local'].fillna('Unknown'))
        except:
            df['type_encoded'] = 0
            
        df['log_surface'] = np.log1p(df['surface_reelle_bati'].fillna(50))
        df['lat_squared'] = df['latitude'] ** 2
        df['lon_squared'] = df['longitude'] ** 2
        df['lat_lon_interaction'] = df['latitude'] * df['longitude']
        
        X = df[self.feature_names].values
        valid_mask = ~np.isnan(X).any(axis=1)
        
        if valid_mask.sum() == 0:
            return None
            
        X_valid = X[valid_mask]
        predictions = self.model.predict(X_valid)
        prices = np.exp(predictions)
        
        median_price = np.median(prices)
        
        # Hierarchical smoothing
        n = len(prices)
        weight = min(1.0, n / 50)
        
        if parent_estimate and weight < 1.0:
            median_price = weight * median_price + (1 - weight) * parent_estimate
        
        # Confidence intervals
        std = np.std(prices) if len(prices) > 1 else median_price * 0.3
        lower = max(500, median_price - 1.96 * std / np.sqrt(max(1, n)))
        upper = median_price + 1.96 * std / np.sqrt(max(1, n))
        
        # Confidence score
        interval_width = (upper - lower) / median_price if median_price > 0 else 1
        model_score = max(5, 40 - interval_width * 50)
        volume_score = min(40, 5 + n * 0.35)
        freshness_score = 15  # Assume recent data
        confidence = int(min(100, model_score + volume_score + freshness_score))
        
        # Quality label
        if confidence >= 70:
            quality = "High"
        elif confidence >= 50:
            quality = "Medium"
        elif confidence >= 30:
            quality = "Low"
        else:
            quality = "Estimated"
        
        return {
            'median_price': round(median_price, 2),
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2),
            'transaction_count': len(transactions_df),
            'confidence_score': confidence,
            'estimate_quality': quality
        }
    
    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_transactions():
    """Load transactions from database."""
    print("\n" + "="*60)
    print("LOADING TRANSACTIONS")
    print("="*60)
    
    query = """
        SELECT 
            date_mutation, valeur_fonciere, code_postal, code_commune,
            code_departement, type_local, surface_reelle_bati,
            nombre_pieces_principales, id_parcelle, price_per_m2,
            ST_X(geom) as longitude, ST_Y(geom) as latitude
        FROM transactions
        WHERE price_per_m2 IS NOT NULL
          AND price_per_m2 BETWEEN 500 AND 20000
    """
    
    start = time.time()
    df = pd.read_sql(query, engine)
    print(f"✓ Loaded {len(df):,} transactions in {time.time()-start:.1f}s")
    
    return df


def load_boundaries():
    """Load administrative boundaries."""
    print("\n" + "="*60)
    print("LOADING BOUNDARIES")
    print("="*60)
    
    # Try to download if missing
    print("\nChecking/downloading boundary files...")
    download_boundaries()
    
    boundaries = {}
    
    # Check multiple possible locations
    data_dirs = [GEOMETRIES_DIR, '/app/data', '/app/data/geometries', './data', './data/geometries']
    
    # Regions - try multiple file patterns
    region_patterns = ['regions.geojson', 'france-regions.geojson', 'regions.json']
    for data_dir in data_dirs:
        for pattern in region_patterns:
            path = os.path.join(data_dir, pattern)
            if os.path.exists(path):
                try:
                    boundaries['regions'] = gpd.read_file(path)
                    print(f"  Regions: {len(boundaries['regions'])} (from {path})")
                    break
                except Exception as e:
                    print(f"  Warning: Failed to load {path}: {e}")
        if 'regions' in boundaries:
            break
    
    # Departments
    dept_patterns = ['departements.geojson', 'departments.geojson', 'france-departements.geojson', 'departements.json']
    for data_dir in data_dirs:
        for pattern in dept_patterns:
            path = os.path.join(data_dir, pattern)
            if os.path.exists(path):
                try:
                    boundaries['departements'] = gpd.read_file(path)
                    print(f"  Départements: {len(boundaries['departements'])} (from {path})")
                    break
                except Exception as e:
                    print(f"  Warning: Failed to load {path}: {e}")
        if 'departements' in boundaries:
            break
    
    # Communes
    commune_patterns = ['communes.geojson', 'france-communes.geojson', 'communes.json']
    for data_dir in data_dirs:
        for pattern in commune_patterns:
            path = os.path.join(data_dir, pattern)
            if os.path.exists(path):
                try:
                    boundaries['communes'] = gpd.read_file(path)
                    print(f"  Communes: {len(boundaries['communes'])} (from {path})")
                    break
                except Exception as e:
                    print(f"  Warning: Failed to load {path}: {e}")
        if 'communes' in boundaries:
            break
    
    # List what files exist in data directories
    if not boundaries:
        print("\n  Searching for boundary files...")
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                geojson_files = [f for f in files if f.endswith('.geojson') or f.endswith('.json')]
                if geojson_files:
                    print(f"  Files in {data_dir}: {geojson_files[:10]}")
    
    if not boundaries:
        print("\n  ⚠️ No boundary files found!")
        print("  The script will create estimates from transaction data.")
    
    return boundaries


# =============================================================================
# AGGREGATE LEVELS (optimized with groupby)
# =============================================================================

def aggregate_level(transactions_gdf, boundaries_gdf, level_name, ml_model, 
                   code_col, name_col, parent_estimates=None):
    """Aggregate prices for a geographic level using groupby."""
    
    print(f"\n{'='*60}")
    print(f"Aggregating: {level_name.upper()}")
    print("="*60)
    
    # Spatial join
    print("Performing spatial join...")
    joined = gpd.sjoin(transactions_gdf, boundaries_gdf, how='inner', predicate='within')
    print(f"Matched: {len(joined):,} transactions in {len(boundaries_gdf)} zones")
    
    results = []
    total_zones = joined[code_col].nunique()
    
    # Use groupby instead of filtering (much faster)
    for i, (code, group) in enumerate(joined.groupby(code_col, sort=False)):
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1:,}/{total_zones:,} zones processed")
        
        # Get zone geometry
        zone = boundaries_gdf[boundaries_gdf[code_col] == code].iloc[0]
        geometry = zone.geometry
        name = zone[name_col] if name_col in zone else str(code)
        
        # Get parent estimate for smoothing
        parent_est = None
        if parent_estimates:
            dept = str(code)[:2] if len(str(code)) > 2 else str(code)
            parent_est = parent_estimates.get(dept)
        
        # Predict
        stats = ml_model.predict_zone(group, geometry, parent_est)
        
        if stats:
            results.append({
                'level': level_name,
                'code': str(code),
                'name': name,
                'geometry': geometry,
                **stats
            })
    
    print(f"✓ Created {len(results):,} {level_name} zones")
    if results:
        prices = [r['median_price'] for r in results]
        print(f"  Price range: €{min(prices):,.0f} - €{max(prices):,.0f}/m²")
        print(f"  Avg confidence: {np.mean([r['confidence_score'] for r in results]):.0f}/100")
    
    return results


# =============================================================================
# PARCEL PROCESSING (Multiprocessing)
# =============================================================================

def get_available_cadastre_depts():
    """Get list of departments with cadastre data."""
    available = set()
    if os.path.exists(CADASTRE_CACHE_DIR):
        for f in os.listdir(CADASTRE_CACHE_DIR):
            if f.startswith('batiments_') and f.endswith('.geojson'):
                dept = f.replace('batiments_', '').replace('_sampled', '').replace('.geojson', '')
                available.add(dept)
    return available


def process_department_parcels(args):
    """
    Process all parcels for a single department.
    This function runs in a separate process.
    """
    dept, dept_df, commune_estimates, model_path = args
    
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, box
    from shapely import affinity
    import joblib
    import numpy as np
    
    # Load model in this process
    ml_model = joblib.load(model_path)
    
    # Load cadastre for this department with spatial index
    cad_gdf = None
    sindex = None
    
    for pattern in [f'batiments_{dept}.geojson', f'batiments_{dept}_sampled.geojson']:
        filepath = f'{CADASTRE_CACHE_DIR}/{pattern}'
        if os.path.exists(filepath):
            try:
                cad_gdf = gpd.read_file(filepath)
                cad_gdf = cad_gdf[cad_gdf.geometry.notna()].copy()
                sindex = cad_gdf.sindex  # Spatial index for fast lookup
            except Exception as e:
                print(f"  Warning: Failed to load cadastre for {dept}: {e}")
            break
    
    results = []
    real_count = 0
    est_count = 0
    
    # Group by parcel ID
    for parcel_id, group in dept_df.groupby('id_parcelle'):
        if len(group) == 0:
            continue
            
        # Get representative point
        lat = group['latitude'].mean()
        lon = group['longitude'].mean()
        point = Point(lon, lat)
        
        # Try to find real building geometry using spatial index
        geometry = None
        
        if cad_gdf is not None and sindex is not None:
            try:
                # Fast contains query
                hits = list(sindex.query(point, predicate="contains"))
                if hits:
                    geometry = cad_gdf.geometry.iloc[hits[0]]
                    real_count += 1
                else:
                    # Try nearest within ~50m
                    candidates = list(sindex.nearest(point, return_all=False))
                    if candidates:
                        idx = candidates[0] if isinstance(candidates[0], int) else candidates[0][1]
                        candidate_geom = cad_gdf.geometry.iloc[idx]
                        if candidate_geom.distance(point) <= 0.0005:  # ~50m
                            geometry = candidate_geom
                            real_count += 1
            except Exception:
                pass
        
        # Create estimated geometry if no real one found
        if geometry is None:
            # Estimate based on surface area
            surface = group['surface_reelle_bati'].mean()
            if pd.isna(surface) or surface <= 0:
                surface = 100
            
            side = np.sqrt(surface) * 0.00001  # Convert m² to degrees approx
            geometry = box(lon - side/2, lat - side/2, lon + side/2, lat + side/2)
            est_count += 1
        
        # Get commune code for parent estimate
        commune_code = group['code_commune'].iloc[0] if 'code_commune' in group else None
        parent_est = commune_estimates.get(commune_code) if commune_code else None
        
        # Predict price
        stats = ml_model.predict_zone(group, geometry, parent_est)
        
        if stats:
            results.append({
                'level': 'parcel',
                'code': str(parcel_id),
                'name': f"Parcel {parcel_id}",
                'geometry': geometry,
                **stats
            })
    
    return {
        'dept': dept,
        'results': results,
        'real': real_count,
        'estimated': est_count
    }


def aggregate_parcels_optimized(transactions_df, ml_model, commune_estimates):
    """Aggregate parcels using spatial index for fast lookup (sequential but optimized)."""
    
    print(f"\n{'='*60}")
    print(f"Aggregating: PARCEL (Optimized with Spatial Index)")
    print("="*60)
    
    # Filter to transactions with parcel IDs
    parcels_df = transactions_df[transactions_df['id_parcelle'].notna()].copy()
    parcels_df['dept_code'] = parcels_df['id_parcelle'].str[:2]
    
    print(f"  Transactions with parcel ID: {len(parcels_df):,}")
    print(f"  Unique parcels: {parcels_df['id_parcelle'].nunique():,}")
    print(f"  Departments: {parcels_df['dept_code'].nunique()}")
    
    # Limit parcels if specified
    if MAX_PARCELS:
        parcel_counts = parcels_df.groupby('id_parcelle').size()
        top_parcels = parcel_counts.nlargest(MAX_PARCELS).index
        parcels_df = parcels_df[parcels_df['id_parcelle'].isin(top_parcels)]
        print(f"  Limited to top {MAX_PARCELS:,} parcels")
    
    # Get available cadastre departments
    available_depts = get_available_cadastre_depts()
    print(f"\n  Available cadastre: {len(available_depts)} departments")
    
    # Sort by department for efficient cadastre loading
    parcels_df = parcels_df.sort_values('dept_code')
    
    # Group by parcel
    parcel_groups = list(parcels_df.groupby('id_parcelle'))
    total_parcels = len(parcel_groups)
    
    print(f"\n  Processing {total_parcels:,} parcels...")
    print("-" * 60)
    
    results = []
    total_real = 0
    total_est = 0
    
    # Cadastre cache with spatial index
    cadastre_cache = {'dept': None, 'data': None, 'sindex': None}
    
    start_time = time.time()
    last_print_time = start_time
    
    for i, (parcel_id, group) in enumerate(parcel_groups):
        # Progress update every 10 seconds or 10000 parcels
        current_time = time.time()
        if (i + 1) % 10000 == 0 or current_time - last_print_time > 10:
            elapsed = current_time - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_parcels - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  ... {i+1:,}/{total_parcels:,} ({100*(i+1)/total_parcels:.1f}%) | "
                  f"Real: {total_real:,} | Est: {total_est:,} | "
                  f"{rate:.0f}/s | ETA: {eta:.0f}min")
            last_print_time = current_time
        
        dept = str(parcel_id)[:2]
        
        # Load cadastre for this department if needed
        if cadastre_cache['dept'] != dept:
            cadastre_cache['data'] = None
            cadastre_cache['sindex'] = None
            cadastre_cache['dept'] = dept
            
            if dept in available_depts:
                for pattern in [f'batiments_{dept}.geojson', f'batiments_{dept}_sampled.geojson']:
                    filepath = f'{CADASTRE_CACHE_DIR}/{pattern}'
                    if os.path.exists(filepath):
                        try:
                            gdf = gpd.read_file(filepath)
                            gdf = gdf[gdf.geometry.notna()].copy()
                            cadastre_cache['data'] = gdf
                            cadastre_cache['sindex'] = gdf.sindex
                        except Exception as e:
                            print(f"  Warning: Failed to load {filepath}: {e}")
                        break
        
        # Get representative point
        lat = group['latitude'].mean()
        lon = group['longitude'].mean()
        point = Point(lon, lat)
        
        # Try to find real building geometry using spatial index
        geometry = None
        cad_gdf = cadastre_cache['data']
        sindex = cadastre_cache['sindex']
        
        if cad_gdf is not None and sindex is not None:
            try:
                # Fast contains query
                hits = list(sindex.query(point, predicate="contains"))
                if hits:
                    geometry = cad_gdf.geometry.iloc[hits[0]]
                    total_real += 1
                else:
                    # Try nearest within ~50m
                    try:
                        # Query nearby geometries
                        buffer = point.buffer(0.0005)  # ~50m
                        candidates = list(sindex.query(buffer))
                        if candidates:
                            # Find closest
                            min_dist = float('inf')
                            closest_geom = None
                            for idx in candidates[:10]:  # Check up to 10 candidates
                                geom = cad_gdf.geometry.iloc[idx]
                                dist = geom.distance(point)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_geom = geom
                            
                            if closest_geom and min_dist <= 0.0005:
                                geometry = closest_geom
                                total_real += 1
                    except:
                        pass
            except Exception:
                pass
        
        # Create estimated geometry if no real one found
        if geometry is None:
            surface = group['surface_reelle_bati'].mean()
            if pd.isna(surface) or surface <= 0:
                surface = 100
            
            side = np.sqrt(surface) * 0.00001
            geometry = box(lon - side/2, lat - side/2, lon + side/2, lat + side/2)
            total_est += 1
        
        # Get commune code for parent estimate
        commune_code = group['code_commune'].iloc[0] if 'code_commune' in group.columns else None
        parent_est = commune_estimates.get(commune_code) if commune_code else None
        
        # Predict price
        stats = ml_model.predict_zone(group, geometry, parent_est)
        
        if stats:
            results.append({
                'level': 'parcel',
                'code': str(parcel_id),
                'name': f"Parcel {parcel_id}",
                'geometry': geometry,
                **stats
            })
    
    elapsed = time.time() - start_time
    
    print("-" * 60)
    print(f"✓ PARCEL AGGREGATION COMPLETE")
    print(f"  Total parcels: {len(results):,}")
    print(f"  With REAL footprints: {total_real:,} ({100*total_real/max(1,len(results)):.1f}%)")
    print(f"  With estimated: {total_est:,} ({100*total_est/max(1,len(results)):.1f}%)")
    print(f"  Time: {elapsed/60:.1f} minutes ({len(results)/max(1,elapsed):.0f} parcels/sec)")
    
    if results:
        prices = [r['median_price'] for r in results]
        print(f"  Price range: €{min(prices):,.0f} - €{max(prices):,.0f}/m²")
    
    return results


# =============================================================================
# SAVE TO DATABASE
# =============================================================================

def save_to_database(results, clear_level=None):
    """Save results to PostgreSQL."""
    if not results:
        return
        
    print(f"\nSaving {len(results):,} aggregates to PostgreSQL...")
    
    with engine.connect() as conn:
        # Clear existing data for this level
        if clear_level:
            conn.execute(text(f"DELETE FROM price_aggregates WHERE level = '{clear_level}'"))
            conn.commit()
        
        # Check if table has geometry column constraint - alter if needed
        try:
            conn.execute(text("""
                ALTER TABLE price_aggregates 
                ALTER COLUMN geom TYPE geometry(Geometry, 4326)
            """))
            conn.commit()
        except:
            conn.rollback()
        
        # Insert in batches
        batch_size = 5000
        inserted = 0
        
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]
            
            for row in batch:
                geom = row.get('geometry')
                if geom is None:
                    geom_wkt = None
                else:
                    # Convert to MultiPolygon if it's a Polygon
                    if geom.geom_type == 'Polygon':
                        geom = MultiPolygon([geom])
                    elif geom.geom_type == 'Point':
                        # Buffer points to create small polygons
                        geom = MultiPolygon([geom.buffer(0.001)])
                    elif geom.geom_type == 'GeometryCollection':
                        # Extract polygons from collection
                        polys = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                        if polys:
                            geom = MultiPolygon([p if p.geom_type == 'Polygon' else list(p.geoms)[0] for p in polys])
                        else:
                            geom_wkt = None
                            continue
                    
                    geom_wkt = geom.wkt if geom else None
                
                # Skip rows with invalid code
                if not row.get('code'):
                    continue
                
                try:
                    conn.execute(text("""
                        INSERT INTO price_aggregates 
                        (level, code, name, median_price, lower_bound, upper_bound,
                         transaction_count, confidence_score, estimate_quality, geom)
                        VALUES (:level, :code, :name, :median_price, :lower_bound, :upper_bound,
                                :transaction_count, :confidence_score, :estimate_quality,
                                ST_GeomFromText(:geom, 4326))
                    """), {
                        'level': row['level'],
                        'code': str(row['code']),
                        'name': row['name'],
                        'median_price': row['median_price'],
                        'lower_bound': row['lower_bound'],
                        'upper_bound': row['upper_bound'],
                        'transaction_count': row['transaction_count'],
                        'confidence_score': row['confidence_score'],
                        'estimate_quality': row['estimate_quality'],
                        'geom': geom_wkt
                    })
                    inserted += 1
                except Exception as e:
                    print(f"  Warning: Failed to insert {row['code']}: {str(e)[:50]}")
            
            conn.commit()
    
    print(f"✓ Saved {inserted:,} aggregates successfully!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("   OPTIMIZED ML PRICE AGGREGATION (Full Cadastre)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {N_WORKERS}")
    print(f"Prediction target: {datetime(2026, 1, 1).date()}")
    
    start_time = time.time()
    
    # Load data
    transactions_df = load_transactions()
    boundaries = load_boundaries()
    
    # Create GeoDataFrame
    transactions_gdf = gpd.GeoDataFrame(
        transactions_df,
        geometry=gpd.points_from_xy(transactions_df['longitude'], transactions_df['latitude']),
        crs="EPSG:4326"
    )
    
    # Train ML model
    print("\n" + "="*60)
    print("TRAINING ML MODEL")
    print("="*60)
    
    ml_model = RobustPriceModel()
    ml_model.fit(transactions_df)
    ml_model.save(MODEL_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
    
    # Clear existing aggregates
    print("\n" + "="*60)
    print("CLEARING EXISTING AGGREGATES")
    print("="*60)
    
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM price_aggregates"))
        conn.commit()
    print("✓ Cleared all existing aggregates")
    
    # Initialize estimate dictionaries
    region_estimates = {}
    dept_estimates = {}
    commune_estimates = {}
    
    # Aggregate each level
    all_results = []
    
    # Country
    country_stats = ml_model.predict_zone(transactions_df)
    if country_stats:
        # Get France geometry
        france_geom = boundaries['regions'].unary_union if 'regions' in boundaries else None
        all_results.append({
            'level': 'country',
            'code': 'FR',
            'name': 'France',
            'geometry': france_geom,
            **country_stats
        })
        print(f"\n✓ France: €{country_stats['median_price']:,.0f}/m²")
    save_to_database(all_results, 'country')
    
    # Regions
    if 'regions' in boundaries:
        region_results = aggregate_level(
            transactions_gdf, boundaries['regions'], 'region',
            ml_model, 'code', 'nom'
        )
        save_to_database(region_results, 'region')
        region_estimates = {r['code']: r['median_price'] for r in region_results}
    else:
        print("\n⚠️ Skipping regions (no boundary file)")
    
    # Departments
    if 'departements' in boundaries:
        dept_results = aggregate_level(
            transactions_gdf, boundaries['departements'], 'departement',
            ml_model, 'code', 'nom', region_estimates
        )
        save_to_database(dept_results, 'departement')
        dept_estimates = {r['code']: r['median_price'] for r in dept_results}
    else:
        # Create dept estimates from transactions
        print("\n⚠️ No department boundaries, creating estimates from transactions...")
        for dept, group in transactions_df.groupby('code_departement'):
            if len(group) >= 10:
                dept_estimates[dept] = group['price_per_m2'].median()
    
    # Communes
    if 'communes' in boundaries:
        commune_results = aggregate_level(
            transactions_gdf, boundaries['communes'], 'commune',
            ml_model, 'code', 'nom', dept_estimates
        )
        save_to_database(commune_results, 'commune')
        commune_estimates = {r['code']: r['median_price'] for r in commune_results}
    else:
        # Create commune estimates from transactions
        print("\n⚠️ No commune boundaries, creating estimates from transactions...")
        for commune, group in transactions_df.groupby('code_commune'):
            if len(group) >= 5:
                commune_estimates[commune] = group['price_per_m2'].median()
    
    # Postcodes (simplified)
    print(f"\n{'='*60}")
    print("Aggregating: POSTCODE")
    print("="*60)
    
    postcode_results = []
    for postcode, group in transactions_df.groupby('code_postal'):
        # Skip invalid postcodes
        if pd.isna(postcode) or str(postcode).strip() == '' or len(group) < 3:
            continue
        
        postcode_str = str(postcode).strip()
        if not postcode_str:
            continue
        
        # Get parent estimate
        dept = postcode_str[:2]
        parent_est = dept_estimates.get(dept)
        
        stats = ml_model.predict_zone(group, None, parent_est)
        if stats:
            # Create convex hull from points
            points = gpd.points_from_xy(group['longitude'], group['latitude'])
            try:
                if len(points) >= 3:
                    geometry = gpd.GeoSeries(points).unary_union.convex_hull
                else:
                    geometry = points[0].buffer(0.005)
                
                # Ensure it's a valid polygon
                if geometry.is_empty or not geometry.is_valid:
                    continue
                    
            except Exception:
                continue
            
            postcode_results.append({
                'level': 'postcode',
                'code': postcode_str,
                'name': f"Postcode {postcode_str}",
                'geometry': geometry,
                **stats
            })
    
    print(f"✓ Created {len(postcode_results):,} postcode zones")
    save_to_database(postcode_results, 'postcode')
    
    # Parcels
    if not SKIP_PARCELS:
        parcel_results = aggregate_parcels_optimized(
            transactions_df, ml_model, commune_estimates
        )
        save_to_database(parcel_results, 'parcel')
    else:
        print("\n⚠️ Skipping parcel aggregation (SKIP_PARCELS=true)")
    
    # Final verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT level, COUNT(*) as zones,
                   ROUND(AVG(median_price)) as avg_price,
                   ROUND(AVG(confidence_score)) as avg_conf,
                   SUM(transaction_count) as total_tx
            FROM price_aggregates
            GROUP BY level
            ORDER BY CASE level
                WHEN 'country' THEN 1 WHEN 'region' THEN 2
                WHEN 'departement' THEN 3 WHEN 'commune' THEN 4
                WHEN 'postcode' THEN 5 WHEN 'parcel' THEN 6
            END
        """))
        
        print("\n" + "-"*70)
        print(f"{'Level':<12} | {'Zones':>10} | {'Avg Price':>14} | {'Confidence':>10} | {'Transactions':>12}")
        print("-"*70)
        
        for row in result:
            print(f"{row[0]:<12} | {row[1]:>10,} | €{row[2]:>10,.0f}/m² | {row[3]:>6}/100 | {row[4]:>12,}")
        print("-"*70)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✓ ML AGGREGATION COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nNext steps:")
    print(f"  1. Export to JSON: python export_to_json.py")
    print(f"  2. Or restart webapp: docker-compose restart webapp")
    print("="*70)


if __name__ == "__main__":
    main()
