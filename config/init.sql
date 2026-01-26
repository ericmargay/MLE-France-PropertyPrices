-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create transactions table with spatial index
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    valeur_fonciere NUMERIC,
    date_mutation DATE,
    type_local VARCHAR(50),
    surface_reelle_bati NUMERIC,
    code_commune VARCHAR(10),
    code_postal VARCHAR(10),
    code_departement VARCHAR(5),
    commune_nom VARCHAR(200),
    price_per_m2 NUMERIC,
    geometry GEOMETRY(Point, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create spatial index for better query performance
CREATE INDEX IF NOT EXISTS idx_transactions_geom 
    ON transactions USING GIST(geometry);

-- Create indexes on frequently queried columns
CREATE INDEX IF NOT EXISTS idx_transactions_commune 
    ON transactions(code_commune);
CREATE INDEX IF NOT EXISTS idx_transactions_postal 
    ON transactions(code_postal);
CREATE INDEX IF NOT EXISTS idx_transactions_departement 
    ON transactions(code_departement);
CREATE INDEX IF NOT EXISTS idx_transactions_date 
    ON transactions(date_mutation);
CREATE INDEX IF NOT EXISTS idx_transactions_type 
    ON transactions(type_local);

-- Create aggregated results table
CREATE TABLE IF NOT EXISTS price_aggregates (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    code VARCHAR(20) NOT NULL,
    name VARCHAR(255),
    median_price DOUBLE PRECISION,
    weighted_price DOUBLE PRECISION,
    std_dev DOUBLE PRECISION,
    transaction_count INTEGER,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    last_transaction_date TIMESTAMP,
    confidence_score INTEGER,
    estimate_quality VARCHAR(20),
    geom GEOMETRY(MULTIPOLYGON, 4326),
    UNIQUE(level, code)
);

-- Create spatial index for aggregates
CREATE INDEX IF NOT EXISTS idx_aggregates_geom 
    ON price_aggregates USING GIST(geom);

-- Create index on level for zoom-based queries
CREATE INDEX IF NOT EXISTS idx_aggregates_level 
    ON price_aggregates(level);

-- Create index on price for color mapping
CREATE INDEX IF NOT EXISTS idx_aggregates_price 
    ON price_aggregates(median_price);

-- Create view for top cities
CREATE OR REPLACE VIEW top_cities_by_volume AS
SELECT 
    code_commune,
    commune_nom,
    type_local,
    COUNT(*) as transaction_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_m2) as median_price,
    AVG(price_per_m2) as avg_price,
    STDDEV(price_per_m2) as std_dev
FROM transactions
WHERE price_per_m2 IS NOT NULL
GROUP BY code_commune, commune_nom, type_local
ORDER BY transaction_count DESC;

-- Create view for price statistics by level
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
    END;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;