"""
Generate top 10 cities analysis by property type
"""
import pandas as pd
from sqlalchemy import create_engine, text
import os
from datetime import datetime

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL',
                         'postgresql://admin:changeme@postgres:5432/property_prices')
engine = create_engine(DATABASE_URL)

def get_top_10_cities():
    """
    Generate report of top 10 cities by transaction volume
    with median prices by property type
    """
    print(f"\n{'='*60}")
    print(f"TOP 10 CITIES ANALYSIS")
    print(f"{'='*60}\n")
    
    query = text("""
        WITH city_transactions AS (
            SELECT 
                code_commune,
                code_commune as city_name,  -- Using code as fallback
                type_local as property_type,
                COUNT(*) as transaction_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_m2) as median_price,
                AVG(price_per_m2) as avg_price,
                STDDEV(price_per_m2) as std_dev,
                MIN(price_per_m2) as min_price,
                MAX(price_per_m2) as max_price
            FROM transactions
            WHERE type_local IN ('Maison', 'Appartement')
                AND price_per_m2 IS NOT NULL
            GROUP BY code_commune, type_local
        ),
        top_cities AS (
            SELECT DISTINCT 
                code_commune, 
                city_name,
                SUM(transaction_count) as total_transactions
            FROM city_transactions
            GROUP BY code_commune, city_name
            ORDER BY total_transactions DESC
            LIMIT 10
        )
        SELECT 
            tc.city_name as "City",
            ct.property_type as "Property Type",
            ROUND(ct.median_price::numeric, 2) as "Median Price (€/m²)",
            ROUND(ct.avg_price::numeric, 2) as "Average Price (€/m²)",
            ROUND(ct.std_dev::numeric, 2) as "Std Dev (€/m²)",
            ct.transaction_count as "Transaction Count",
            ROUND(ct.min_price::numeric, 2) as "Min Price (€/m²)",
            ROUND(ct.max_price::numeric, 2) as "Max Price (€/m²)"
        FROM top_cities tc
        JOIN city_transactions ct 
            ON tc.code_commune = ct.code_commune
        ORDER BY tc.total_transactions DESC, ct.property_type
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Generated report for {len(df['City'].unique())} cities")
    print(f"Total rows: {len(df)}")
    
    return df

def save_report(df, output_path='/app/results/top_10_cities.csv'):
    """Save report to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Report saved to: {output_path}")
    return output_path

def print_summary(df):
    """Print formatted summary"""
    print(f"\n{'='*80}")
    print(f"TOP 10 CITIES BY TRANSACTION VOLUME - PRICE ANALYSIS")
    print(f"{'='*80}\n")
    
    for city in df['City'].unique():
        city_data = df[df['City'] == city]
        total_trans = city_data['Transaction Count'].sum()
        
        print(f"\n{city} (Total: {total_trans:,} transactions)")
        print("-" * 80)
        
        for _, row in city_data.iterrows():
            print(f"  {row['Property Type']:12} | "
                  f"Median: €{row['Median Price (€/m²)']:7,.0f}/m² | "
                  f"Avg: €{row['Average Price (€/m²)']:7,.0f}/m² | "
                  f"Count: {row['Transaction Count']:6,}")
    
    print("\n" + "="*80)

def main():
    """Main execution"""
    print(f"\nStarted at: {datetime.now()}\n")
    
    try:
        # Generate report
        df = get_top_10_cities()
        
        # Print summary
        print_summary(df)
        
        # Save to CSV
        output_path = save_report(df)
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"Finished at: {datetime.now()}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)