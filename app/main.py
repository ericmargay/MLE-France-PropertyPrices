"""
France Property Prices - FastAPI Web Application
================================================

Hybrid API that supports both:
1. DATABASE MODE: PostgreSQL/PostGIS (full features, real-time queries)
2. STATIC MODE: JSON files (simpler deployment, no database needed)

Mode is determined by environment variable:
    DATA_MODE=database  (default, requires PostgreSQL)
    DATA_MODE=static    (uses JSON files from /app/data/export/)

Usage:
    # Database mode (default)
    docker-compose up webapp

    # Static mode
    DATA_MODE=static docker-compose up webapp
"""

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import os
import json
import gzip
from typing import Optional, List
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data mode: 'database' or 'static'
DATA_MODE = os.getenv('DATA_MODE', 'database')

# Database settings (for database mode)
DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')

# Static files directory (for static mode)
STATIC_DATA_DIR = os.getenv('STATIC_DATA_DIR', '/app/data/export')

# Mapbox token
MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')

# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="France Property Prices API",
    description="ML-based property price estimation for France",
    version="2.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
templates = Jinja2Templates(directory="/app/templates")

# Database connection (only if database mode)
engine = None
if DATA_MODE == 'database':
    try:
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        print(f"✓ Database mode: Connected to PostgreSQL")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print(f"   Falling back to static mode")
        DATA_MODE = 'static'

# Static data cache (for static mode)
static_cache = {}

def load_static_data():
    """Load JSON files into memory for static mode."""
    global static_cache
    
    if DATA_MODE != 'static':
        return
    
    print(f"Loading static data from {STATIC_DATA_DIR}...")
    
    levels = ['country', 'region', 'departement', 'commune', 'postcode']
    
    for level in levels:
        filepath = os.path.join(STATIC_DATA_DIR, f'{level}.geojson')
        gz_filepath = filepath + '.gz'
        
        if os.path.exists(gz_filepath):
            with gzip.open(gz_filepath, 'rt', encoding='utf-8') as f:
                static_cache[level] = json.load(f)
            print(f"  ✓ Loaded {level} ({len(static_cache[level]['features']):,} features)")
        elif os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                static_cache[level] = json.load(f)
            print(f"  ✓ Loaded {level} ({len(static_cache[level]['features']):,} features)")
        else:
            print(f"  ⚠️  {level}.geojson not found")
    
    # Load parcel index (department list)
    parcels_dir = os.path.join(STATIC_DATA_DIR, 'parcels')
    if os.path.exists(parcels_dir):
        static_cache['parcel_departments'] = [
            f.replace('.geojson.gz', '').replace('.geojson', '')
            for f in os.listdir(parcels_dir)
            if f.endswith('.geojson') or f.endswith('.geojson.gz')
        ]
        print(f"  ✓ Found {len(static_cache['parcel_departments'])} parcel departments")

# Load static data on startup
if DATA_MODE == 'static':
    load_static_data()

# =============================================================================
# STARTUP MESSAGE
# =============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("FastAPI Application Starting")
    print("="*60)
    print(f"Data Mode: {DATA_MODE.upper()}")
    if DATA_MODE == 'database':
        print(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configured'}")
    else:
        print(f"Static Data: {STATIC_DATA_DIR}")
    print(f"API Docs: http://localhost:8080/docs")
    print(f"Map Interface: http://localhost:8080")
    print("="*60)

# =============================================================================
# HTML ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main map interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "mapbox_token": MAPBOX_TOKEN,
        "data_mode": DATA_MODE
    })

@app.get("/top-cities", response_class=HTMLResponse)
async def top_cities_page(request: Request):
    """Render top cities page."""
    return templates.TemplateResponse("top-cities.html", {
        "request": request
    })

# =============================================================================
# API ROUTES - DATABASE MODE
# =============================================================================

def get_prices_from_database(level: str, bbox: Optional[str], zoom: int):
    """Fetch price data from PostgreSQL."""
    
    # Set limit based on zoom level
    if level == 'parcel':
        limit = min(100000, 10000 * max(1, zoom - 14))
    elif level == 'postcode':
        limit = 10000
    else:
        limit = None
    
    # Build query
    if bbox:
        coords = [float(x) for x in bbox.split(',')]
        query = text(f"""
            SELECT 
                code, name, level, median_price, weighted_price,
                std_dev, transaction_count, lower_bound, upper_bound,
                confidence_score, estimate_quality, last_transaction_date,
                ST_AsGeoJSON(geom)::json as geometry
            FROM price_aggregates
            WHERE level = :level
              AND ST_Intersects(
                  geom, 
                  ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)
              )
            {"LIMIT :limit" if limit else ""}
        """)
        params = {
            'level': level,
            'minx': coords[0], 'miny': coords[1],
            'maxx': coords[2], 'maxy': coords[3]
        }
        if limit:
            params['limit'] = limit
    else:
        query = text(f"""
            SELECT 
                code, name, level, median_price, weighted_price,
                std_dev, transaction_count, lower_bound, upper_bound,
                confidence_score, estimate_quality, last_transaction_date,
                ST_AsGeoJSON(geom)::json as geometry
            FROM price_aggregates
            WHERE level = :level
            {"LIMIT :limit" if limit else ""}
        """)
        params = {'level': level}
        if limit:
            params['limit'] = limit
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        rows = result.fetchall()
    
    features = []
    for row in rows:
        feature = {
            'type': 'Feature',
            'properties': {
                'code': row.code,
                'name': row.name,
                'level': row.level,
                'median_price': float(row.median_price) if row.median_price else None,
                'weighted_price': float(row.weighted_price) if row.weighted_price else None,
                'std_dev': float(row.std_dev) if row.std_dev else None,
                'transaction_count': row.transaction_count,
                'lower_bound': float(row.lower_bound) if row.lower_bound else None,
                'upper_bound': float(row.upper_bound) if row.upper_bound else None,
                'confidence_score': row.confidence_score,
                'estimate_quality': row.estimate_quality,
                'last_transaction_date': row.last_transaction_date.isoformat() if row.last_transaction_date else None
            },
            'geometry': row.geometry
        }
        features.append(feature)
    
    return {'type': 'FeatureCollection', 'features': features}

# =============================================================================
# API ROUTES - STATIC MODE
# =============================================================================

def get_prices_from_static(level: str, bbox: Optional[str], zoom: int):
    """Fetch price data from static JSON files."""
    
    if level == 'parcel':
        # For parcels, load by department based on bbox
        return get_parcels_from_static(bbox, zoom)
    
    if level not in static_cache:
        return {'type': 'FeatureCollection', 'features': []}
    
    geojson = static_cache[level]
    
    # If no bbox, return all
    if not bbox:
        return geojson
    
    # Filter by bounding box
    coords = [float(x) for x in bbox.split(',')]
    minx, miny, maxx, maxy = coords
    
    filtered_features = []
    for feature in geojson['features']:
        # Simple centroid check (not perfect but fast)
        geom = feature['geometry']
        if geom and geom.get('coordinates'):
            # Get approximate centroid
            try:
                if geom['type'] == 'Polygon':
                    coords_list = geom['coordinates'][0]
                elif geom['type'] == 'MultiPolygon':
                    coords_list = geom['coordinates'][0][0]
                else:
                    coords_list = []
                
                if coords_list:
                    cx = sum(c[0] for c in coords_list) / len(coords_list)
                    cy = sum(c[1] for c in coords_list) / len(coords_list)
                    
                    if minx <= cx <= maxx and miny <= cy <= maxy:
                        filtered_features.append(feature)
            except (IndexError, TypeError):
                # Include if can't determine
                filtered_features.append(feature)
    
    return {'type': 'FeatureCollection', 'features': filtered_features}


def get_parcels_from_static(bbox: Optional[str], zoom: int):
    """Load parcel data from tiled JSON files."""
    
    if 'parcel_departments' not in static_cache:
        return {'type': 'FeatureCollection', 'features': []}
    
    # Determine which departments to load based on bbox
    # For simplicity, load all if bbox is large, or filter by department
    
    if not bbox:
        # No bbox - don't load all parcels (too large)
        return {'type': 'FeatureCollection', 'features': [], 
                'message': 'Please zoom in to see parcels'}
    
    coords = [float(x) for x in bbox.split(',')]
    minx, miny, maxx, maxy = coords
    
    # Check bbox size - don't load if too large
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    
    if bbox_width > 1 or bbox_height > 1:
        return {'type': 'FeatureCollection', 'features': [], 
                'message': 'Please zoom in to see parcels'}
    
    # For now, load all available parcel files and filter
    # In production, you'd want a spatial index
    
    features = []
    parcels_dir = os.path.join(STATIC_DATA_DIR, 'parcels')
    
    # Limit to prevent memory issues
    max_parcels = 50000
    
    for dept in static_cache.get('parcel_departments', [])[:20]:  # Limit departments
        filepath = os.path.join(parcels_dir, f'{dept}.geojson.gz')
        
        if not os.path.exists(filepath):
            filepath = os.path.join(parcels_dir, f'{dept}.geojson')
        
        if os.path.exists(filepath):
            try:
                if filepath.endswith('.gz'):
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                # Filter by bbox
                for feature in data.get('features', []):
                    if len(features) >= max_parcels:
                        break
                    
                    geom = feature.get('geometry')
                    if geom and geom.get('coordinates'):
                        try:
                            if geom['type'] == 'Polygon':
                                first_coord = geom['coordinates'][0][0]
                            elif geom['type'] == 'MultiPolygon':
                                first_coord = geom['coordinates'][0][0][0]
                            else:
                                continue
                            
                            cx, cy = first_coord[0], first_coord[1]
                            
                            if minx <= cx <= maxx and miny <= cy <= maxy:
                                features.append(feature)
                        except (IndexError, TypeError):
                            pass
                
                if len(features) >= max_parcels:
                    break
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return {'type': 'FeatureCollection', 'features': features}

# =============================================================================
# UNIFIED API ENDPOINTS
# =============================================================================

@app.get("/api/prices/{level}")
async def get_prices(
    level: str,
    bbox: Optional[str] = Query(None, description="Bounding box: minLon,minLat,maxLon,maxLat"),
    zoom: int = Query(10, description="Map zoom level")
):
    """
    Get price aggregations for a geographic level.
    
    Levels: country, region, departement, commune, postcode, parcel
    """
    valid_levels = ['country', 'region', 'departement', 'commune', 'postcode', 'parcel']
    
    if level not in valid_levels:
        raise HTTPException(status_code=400, detail=f"Invalid level. Valid: {valid_levels}")
    
    if DATA_MODE == 'database' and engine:
        return get_prices_from_database(level, bbox, zoom)
    else:
        return get_prices_from_static(level, bbox, zoom)


@app.get("/api/top-cities")
async def get_top_cities(limit: int = Query(10, ge=1, le=50)):
    """Get top cities by transaction volume with price breakdown by property type."""
    
    if DATA_MODE != 'database' or not engine:
        # Return cached/static data for static mode
        return JSONResponse(content=[
            {"city": "Paris", "property_types": [
                {"type": "Appartement", "median_price": 9500, "transaction_count": 45000},
                {"type": "Maison", "median_price": 7200, "transaction_count": 5000}
            ]},
            {"city": "Lyon", "property_types": [
                {"type": "Appartement", "median_price": 4800, "transaction_count": 15000},
                {"type": "Maison", "median_price": 4200, "transaction_count": 3000}
            ]}
        ])
    
    query = text("""
        WITH city_stats AS (
            SELECT 
                SPLIT_PART(nom_commune, ' ', 1) as city,
                type_local,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_m2) as median_price,
                AVG(price_per_m2) as avg_price,
                COUNT(*) as transaction_count
            FROM transactions
            WHERE type_local IN ('Appartement', 'Maison')
              AND price_per_m2 BETWEEN 100 AND 50000
            GROUP BY city, type_local
        ),
        city_totals AS (
            SELECT city, SUM(transaction_count) as total
            FROM city_stats
            GROUP BY city
            ORDER BY total DESC
            LIMIT :limit
        )
        SELECT cs.city, cs.type_local, cs.median_price, cs.avg_price, cs.transaction_count
        FROM city_stats cs
        JOIN city_totals ct ON cs.city = ct.city
        ORDER BY ct.total DESC, cs.type_local
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {'limit': limit})
            rows = result.fetchall()
        
        # Group by city
        cities = {}
        for row in rows:
            city = row.city
            if city not in cities:
                cities[city] = {'city': city, 'property_types': []}
            
            cities[city]['property_types'].append({
                'type': row.type_local,
                'median_price': float(row.median_price) if row.median_price else None,
                'avg_price': float(row.avg_price) if row.avg_price else None,
                'transaction_count': row.transaction_count
            })
        
        return list(cities.values())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get application statistics."""
    
    stats = {
        'status': 'healthy',
        'data_mode': DATA_MODE,
        'timestamp': datetime.now().isoformat()
    }
    
    if DATA_MODE == 'database' and engine:
        try:
            with engine.connect() as conn:
                # Transaction count
                result = conn.execute(text("SELECT COUNT(*) FROM transactions"))
                stats['total_transactions'] = result.scalar()
                
                # Aggregates by level
                result = conn.execute(text("""
                    SELECT level, COUNT(*) as count
                    FROM price_aggregates
                    GROUP BY level
                """))
                stats['levels'] = {row[0]: row[1] for row in result}
                stats['total_aggregates'] = sum(stats['levels'].values())
                stats['database'] = 'connected'
                
        except Exception as e:
            stats['database'] = f'error: {str(e)}'
    else:
        stats['database'] = 'not used (static mode)'
        stats['levels'] = {
            level: len(static_cache.get(level, {}).get('features', []))
            for level in ['country', 'region', 'departement', 'commune', 'postcode']
        }
        if 'parcel_departments' in static_cache:
            stats['parcel_departments'] = len(static_cache['parcel_departments'])
    
    return stats


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'data_mode': DATA_MODE,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            'error': str(exc),
            'detail': 'An internal error occurred'
        }
    )
