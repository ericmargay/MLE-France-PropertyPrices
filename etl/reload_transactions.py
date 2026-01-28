"""
Quick script to reload transactions into database.
Run this if the transactions table is empty.

Usage: docker-compose run --rm etl python reload_transactions.py
"""

import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from shapely.geometry import Point
import os
import requests
import gzip

DATABASE_URL = os.getenv('DATABASE_URL', 
                         'postgresql://admin:changeme@postgres:5432/property_prices')
engine = create_engine(DATABASE_URL)

DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/full.csv.gz"
DATA_DIR = "/app/data"


def setup_table():
    """Ensure transactions table has correct schema."""
    print("Setting up transactions table...")
    
    with engine.connect() as conn:
        conn.execute(text("""
            DROP TABLE IF EXISTS transactions CASCADE;
            
            CREATE TABLE transactions (
                id SERIAL PRIMARY KEY,
                date_mutation DATE,
                nature_mutation VARCHAR(50),
                valeur_fonciere NUMERIC,
                code_postal VARCHAR(10),
                code_commune VARCHAR(10),
                code_departement VARCHAR(5),
                type_local VARCHAR(50),
                surface_reelle_bati NUMERIC,
                nombre_pieces_principales INTEGER,
                id_parcelle VARCHAR(50),
                price_per_m2 NUMERIC,
                geom GEOMETRY(Point, 4326)
            );
            
            CREATE INDEX idx_transactions_geom ON transactions USING GIST(geom);
            CREATE INDEX idx_transactions_commune ON transactions(code_commune);
            CREATE INDEX idx_transactions_dept ON transactions(code_departement);
            CREATE INDEX idx_transactions_price ON transactions(price_per_m2);
        """))
        conn.commit()
    
    print("✓ Table created")


def download_dvf():
    """Download DVF data if not cached."""
    csv_path = f"{DATA_DIR}/dvf_2023.csv"
    gz_path = f"{DATA_DIR}/dvf_2023.csv.gz"
    
    if os.path.exists(csv_path):
        print(f"✓ Using cached: {csv_path}")
        return csv_path
    
    print(f"Downloading DVF 2023 data...")
    response = requests.get(DVF_URL, stream=True)
    
    with open(gz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("  Decompressing...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(csv_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    os.remove(gz_path)
    print(f"✓ Downloaded: {csv_path}")
    return csv_path


def load_and_clean(csv_path):
    """Load and clean DVF data."""
    print("Loading CSV...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")
    
    # Filter to residential properties with valid data
    df = df[df['type_local'].isin(['Maison', 'Appartement'])]
    df = df[df['valeur_fonciere'].notna()]
    df = df[df['surface_reelle_bati'].notna()]
    df = df[df['surface_reelle_bati'] > 0]
    df = df[df['longitude'].notna()]
    df = df[df['latitude'].notna()]
    
    # Calculate price per m2
    df['price_per_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    
    # Filter outliers
    df = df[(df['price_per_m2'] >= 500) & (df['price_per_m2'] <= 20000)]
    
    # Filter France bounds
    df = df[(df['longitude'] >= -5.5) & (df['longitude'] <= 10)]
    df = df[(df['latitude'] >= 41) & (df['latitude'] <= 51.5)]
    
    print(f"  Cleaned rows: {len(df):,}")
    
    return df


def save_to_db(df):
    """Save transactions to database using raw SQL for geometry."""
    print(f"Saving {len(df):,} transactions to database...")
    
    # Prepare data
    df_save = df[[
        'date_mutation', 'nature_mutation', 'valeur_fonciere',
        'code_postal', 'code_commune', 'code_departement',
        'type_local', 'surface_reelle_bati', 'nombre_pieces_principales',
        'id_parcelle', 'price_per_m2', 'longitude', 'latitude'
    ]].copy()
    
    # Clean data
    df_save['nombre_pieces_principales'] = df_save['nombre_pieces_principales'].fillna(0).astype(int)
    df_save = df_save.fillna('')
    
    # Insert in chunks using raw SQL
    chunk_size = 5000
    total_chunks = (len(df_save) + chunk_size - 1) // chunk_size
    total_inserted = 0
    
    with engine.connect() as conn:
        for i in range(0, len(df_save), chunk_size):
            chunk = df_save.iloc[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            if chunk_num % 10 == 0 or chunk_num == 1:
                print(f"  ... chunk {chunk_num}/{total_chunks} ({total_inserted:,} rows)")
            
            # Build VALUES clause
            values_list = []
            for _, row in chunk.iterrows():
                # Escape single quotes in strings
                nature = str(row['nature_mutation']).replace("'", "''") if row['nature_mutation'] else ''
                type_local = str(row['type_local']).replace("'", "''") if row['type_local'] else ''
                id_parcelle = str(row['id_parcelle']).replace("'", "''") if row['id_parcelle'] else ''
                
                values_list.append(f"""(
                    '{row['date_mutation']}',
                    '{nature}',
                    {row['valeur_fonciere']},
                    '{row['code_postal']}',
                    '{row['code_commune']}',
                    '{row['code_departement']}',
                    '{type_local}',
                    {row['surface_reelle_bati']},
                    {row['nombre_pieces_principales']},
                    '{id_parcelle}',
                    {row['price_per_m2']},
                    ST_SetSRID(ST_MakePoint({row['longitude']}, {row['latitude']}), 4326)
                )""")
            
            sql = f"""
                INSERT INTO transactions (
                    date_mutation, nature_mutation, valeur_fonciere,
                    code_postal, code_commune, code_departement,
                    type_local, surface_reelle_bati, nombre_pieces_principales,
                    id_parcelle, price_per_m2, geom
                ) VALUES {','.join(values_list)}
            """
            
            conn.execute(text(sql))
            conn.commit()
            total_inserted += len(chunk)
    
    print(f"✓ Saved {total_inserted:,} transactions")


def main():
    print("\n" + "="*60)
    print("RELOADING TRANSACTIONS")
    print("="*60)
    
    # Setup table with correct schema
    setup_table()
    
    # Download DVF data
    csv_path = download_dvf()
    
    # Load and clean
    df = load_and_clean(csv_path)
    
    # Save to database
    save_to_db(df)
    
    # Verify
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
    
    print("\n" + "="*60)
    print(f"✓ COMPLETE: {count:,} transactions loaded")
    print("="*60)
    print("\nNow run: docker-compose run --rm etl python regenerate_with_ml.py")


if __name__ == "__main__":
    main()
