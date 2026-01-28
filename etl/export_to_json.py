"""
Export Price Aggregations to GeoJSON Files
==========================================

Exports all aggregations to static GeoJSON files for deployment
without a database. Perfect for hosting on Vercel, Netlify, or GitHub Pages.

Output:
    /app/static/data/
    ├── country.json        (~1 KB)
    ├── region.json         (~50 KB)  
    ├── departement.json    (~200 KB)
    ├── commune.json        (~15 MB, compressed)
    ├── postcode.json       (~5 MB)
    ├── parcel/             (split by department)
    │   ├── 01.json, 02.json, ...
    ├── parcel_index.json   (department listing)
    └── metadata.json       (stats)

Usage:
    docker-compose run --rm etl python export_to_json.py
"""

import os
import json
import gzip
import argparse
from datetime import datetime
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
OUTPUT_DIR = '/app/static/data'
ALL_LEVELS = ['country', 'region', 'departement', 'commune', 'postcode', 'parcel']


def export_level(engine, level, output_dir):
    """Export a single level to GeoJSON."""
    print(f"\n  Exporting {level}...", end=" ", flush=True)
    
    query = text("""
        SELECT code, name, median_price, weighted_price, std_dev,
               transaction_count, lower_bound, upper_bound,
               confidence_score, estimate_quality, last_transaction_date,
               ST_AsGeoJSON(geom)::json as geometry
        FROM price_aggregates WHERE level = :level
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(query, {'level': level}).fetchall()
    
    if not rows:
        print(f"⚠️ No data")
        return {'level': level, 'count': 0, 'size': 0}
    
    features = []
    for row in rows:
        features.append({
            'type': 'Feature',
            'properties': {
                'code': row.code,
                'name': row.name,
                'median_price': float(row.median_price) if row.median_price else None,
                'weighted_price': float(row.weighted_price) if row.weighted_price else None,
                'std_dev': float(row.std_dev) if row.std_dev else None,
                'transaction_count': row.transaction_count,
                'lower_bound': float(row.lower_bound) if row.lower_bound else None,
                'upper_bound': float(row.upper_bound) if row.upper_bound else None,
                'confidence_score': row.confidence_score,
                'estimate_quality': row.estimate_quality,
                'last_transaction_date': str(row.last_transaction_date) if row.last_transaction_date else None
            },
            'geometry': row.geometry
        })
    
    geojson = {'type': 'FeatureCollection', 'features': features}
    
    # Handle parcels differently - split by department
    if level == 'parcel':
        parcel_dir = os.path.join(output_dir, 'parcel')
        os.makedirs(parcel_dir, exist_ok=True)
        
        # Group by department (first 2 chars of code)
        by_dept = {}
        for f in features:
            dept = f['properties']['code'][:2]
            if dept not in by_dept:
                by_dept[dept] = []
            by_dept[dept].append(f)
        
        total_size = 0
        for dept, dept_features in by_dept.items():
            dept_geojson = {'type': 'FeatureCollection', 'features': dept_features}
            filepath = os.path.join(parcel_dir, f'{dept}.json')
            
            with open(filepath, 'w') as f:
                json.dump(dept_geojson, f, separators=(',', ':'))
            total_size += os.path.getsize(filepath)
        
        # Create parcel index
        index = {'departments': sorted(by_dept.keys()), 'total': len(features)}
        with open(os.path.join(output_dir, 'parcel_index.json'), 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"✓ {len(features):,} parcels → {len(by_dept)} dept files ({total_size/1024/1024:.1f} MB)")
        return {'level': level, 'count': len(features), 'size': total_size}
    
    # Regular level - single file
    filepath = os.path.join(output_dir, f'{level}.json')
    
    with open(filepath, 'w') as f:
        json.dump(geojson, f, separators=(',', ':'))
    
    file_size = os.path.getsize(filepath)
    
    # Compress large files
    if file_size > 1024 * 1024:  # > 1MB
        with open(filepath, 'rb') as f_in:
            with gzip.open(filepath + '.gz', 'wb') as f_out:
                f_out.write(f_in.read())
        gz_size = os.path.getsize(filepath + '.gz')
        print(f"✓ {len(features):,} features ({file_size/1024/1024:.1f} MB → {gz_size/1024/1024:.1f} MB gz)")
    else:
        print(f"✓ {len(features):,} features ({file_size/1024:.1f} KB)")
    
    return {'level': level, 'count': len(features), 'size': file_size}


def export_metadata(engine, output_dir, stats):
    """Create metadata file."""
    with engine.connect() as conn:
        tx_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'prediction_date': '2025-01-01',
        'model': 'LightGBM + Linear Temporal Trend',
        'data_source': 'DVF 2023 (data.gouv.fr)',
        'total_transactions': tx_count,
        'levels': {s['level']: s['count'] for s in stats},
        'files': {
            'country': 'country.json',
            'region': 'region.json',
            'departement': 'departement.json',
            'commune': 'commune.json',
            'postcode': 'postcode.json',
            'parcel': 'parcel/{dept}.json'
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  ✓ metadata.json created")


def export_top_cities(engine, output_dir):
    """Export top 10 biggest cities with prices by property type."""
    
    # Top 10 biggest cities by transaction count
    query = """
        WITH city_stats AS (
            SELECT 
                code_commune,
                type_local,
                COUNT(*) as tx_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_m2) as median_price
            FROM transactions
            WHERE type_local IN ('Appartement', 'Maison')
              AND price_per_m2 IS NOT NULL
            GROUP BY code_commune, type_local
        ),
        city_totals AS (
            SELECT 
                code_commune,
                SUM(tx_count) as total_transactions
            FROM city_stats
            GROUP BY code_commune
            ORDER BY total_transactions DESC
            LIMIT 10
        )
        SELECT 
            pa.name,
            pa.code,
            cs.type_local,
            cs.median_price,
            cs.tx_count
        FROM city_totals ct
        JOIN price_aggregates pa ON pa.code = ct.code_commune AND pa.level = 'commune'
        JOIN city_stats cs ON cs.code_commune = ct.code_commune
        ORDER BY ct.total_transactions DESC, cs.type_local
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
    
    # Group by city
    cities_dict = {}
    for row in rows:
        name, code, prop_type, median_price, tx_count = row
        
        if code not in cities_dict:
            cities_dict[code] = {
                'name': name,
                'code': code,
                'property_types': []
            }
        
        cities_dict[code]['property_types'].append({
            'type': prop_type,
            'median_price': round(float(median_price)) if median_price else None,
            'transaction_count': int(tx_count) if tx_count else 0
        })
    
    cities = list(cities_dict.values())
    
    filepath = os.path.join(output_dir, 'top_cities.json')
    with open(filepath, 'w') as f:
        json.dump(cities, f, indent=2)
    
    print(f"  ✓ top_cities.json: {len(cities)} cities with property types")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', default=','.join(ALL_LEVELS))
    parser.add_argument('--output', default=OUTPUT_DIR)
    args = parser.parse_args()
    
    levels = [l.strip() for l in args.levels.split(',')]
    
    print("\n" + "="*60)
    print("EXPORTING TO GEOJSON (Static Hosting)")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Levels: {', '.join(levels)}")
    
    os.makedirs(args.output, exist_ok=True)
    engine = create_engine(DATABASE_URL)
    
    stats = []
    for level in levels:
        if level in ALL_LEVELS:
            stats.append(export_level(engine, level, args.output))
    
    export_metadata(engine, args.output, stats)
    export_top_cities(engine, args.output)
    
    total = sum(s['count'] for s in stats)
    total_size = sum(s['size'] for s in stats)
    
    print("\n" + "="*60)
    print(f"✓ EXPORT COMPLETE")
    print(f"  Total: {total:,} features ({total_size/1024/1024:.1f} MB)")
    print(f"  Files in: {args.output}")
    print("="*60)
    print("\nNext: Copy to your deployment")
    print("  docker cp $(docker-compose ps -q etl):/app/static/data ./static/data")


if __name__ == "__main__":
    main()
