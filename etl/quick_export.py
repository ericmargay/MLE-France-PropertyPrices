"""
Quick export of current database data to JSON for testing static web.
Run in a SEPARATE terminal while ETL is still processing parcels.

Usage: docker-compose run --rm etl python quick_export.py
"""

import json
import os
from sqlalchemy import create_engine, text
from datetime import datetime

DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
engine = create_engine(DATABASE_URL)

OUTPUT_DIR = '/app/static/data'


def export_top_cities():
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
    
    filepath = os.path.join(OUTPUT_DIR, 'top_cities.json')
    with open(filepath, 'w') as f:
        json.dump(cities, f, indent=2)
    
    print(f"  ✓ top_cities: {len(cities)} cities with property types")


def export_level(level_name):
    """Export a single level to GeoJSON."""
    
    # Use 'geom' instead of 'geometry' (PostGIS convention)
    query = f"""
        SELECT 
            code,
            name,
            median_price,
            lower_bound,
            upper_bound,
            transaction_count,
            confidence_score,
            estimate_quality,
            ST_AsGeoJSON(geom)::json as geometry
        FROM price_aggregates
        WHERE level = '{level_name}'
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
    
    if not rows:
        print(f"  ⚠️  No data for {level_name}")
        return 0
    
    features = []
    for row in rows:
        features.append({
            'type': 'Feature',
            'properties': {
                'code': row[0],
                'name': row[1],
                'median_price': float(row[2]) if row[2] else None,
                'lower_bound': float(row[3]) if row[3] else None,
                'upper_bound': float(row[4]) if row[4] else None,
                'transaction_count': int(row[5]) if row[5] else 0,
                'confidence_score': int(row[6]) if row[6] else 0,
                'estimate_quality': row[7]
            },
            'geometry': row[8]
        })
    
    geojson = {'type': 'FeatureCollection', 'features': features}
    
    # Handle parcels - split by department
    if level_name == 'parcel' and len(features) > 0:
        parcel_dir = os.path.join(OUTPUT_DIR, 'parcel')
        os.makedirs(parcel_dir, exist_ok=True)
        
        by_dept = {}
        for f in features:
            dept = f['properties']['code'][:2]
            if dept not in by_dept:
                by_dept[dept] = []
            by_dept[dept].append(f)
        
        for dept, dept_features in by_dept.items():
            filepath = os.path.join(parcel_dir, f'{dept}.json')
            with open(filepath, 'w') as f:
                json.dump({'type': 'FeatureCollection', 'features': dept_features}, f, separators=(',', ':'))
        
        # Create index
        index = {'departments': sorted(by_dept.keys()), 'total': len(features)}
        with open(os.path.join(OUTPUT_DIR, 'parcel_index.json'), 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"  ✓ {level_name}: {len(features):,} features → {len(by_dept)} dept files")
        return len(features)
    
    # Regular level - single file
    filepath = os.path.join(OUTPUT_DIR, f'{level_name}.json')
    with open(filepath, 'w') as f:
        json.dump(geojson, f, separators=(',', ':'))
    
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  ✓ {level_name}: {len(features):,} features ({size_kb:.0f} KB)")
    
    return len(features)


def main():
    print("\n" + "="*60)
    print("QUICK EXPORT FOR STATIC WEB TESTING")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check what's in database
    print("\nChecking database...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT level, COUNT(*) as cnt 
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
        levels = {row[0]: row[1] for row in result}
    
    print("  Available data:")
    for level, count in levels.items():
        print(f"    {level}: {count:,}")
    
    # Export each level
    print("\nExporting...")
    
    total = 0
    for level in ['country', 'region', 'departement', 'commune', 'postcode', 'parcel']:
        if level in levels:
            total += export_level(level)
    
    # Create metadata
    metadata = {
        'exported_at': datetime.now().isoformat(),
        'levels': levels,
        'total_features': total,
        'note': 'Partial export for testing (parcels may be incomplete)'
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Export top cities
    print("\nExporting top cities...")
    export_top_cities()
    
    print(f"\n" + "="*60)
    print(f"✓ EXPORT COMPLETE")
    print(f"="*60)
    print(f"Total features: {total:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nTo copy to your machine:")
    print(f"  docker cp $(docker-compose ps -q etl):/app/static/data ./static-test/")
    print(f"\nOr run another container to copy:")
    print(f"  docker-compose run --rm -v $(pwd)/static-test:/output etl cp -r /app/static/data/* /output/")


if __name__ == "__main__":
    main()
