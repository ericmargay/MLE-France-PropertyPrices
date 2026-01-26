#!/usr/bin/env bash
set -euo pipefail

COMPOSE="docker-compose"
DB_SERVICE="postgres"
DB_USER="admin"
DB_NAME="property_prices"

echo "== Full France ETL (process_data.py) =="
echo "Repo: $(pwd)"
echo

# 1) Build ETL image (picks up latest code changes)
echo "== Building ETL image =="
$COMPOSE build etl
echo

# 2) Ensure Postgres is running
echo "== Starting Postgres (if needed) =="
$COMPOSE up -d "$DB_SERVICE"
echo

# 3) Wait for Postgres to be ready
echo "== Waiting for Postgres to be ready =="
until $COMPOSE exec -T "$DB_SERVICE" pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; do
  echo "  ... postgres not ready yet"
  sleep 1
done
echo "✅ Postgres is ready"
echo

# 4) Ensure PostGIS extension exists
echo "== Ensuring PostGIS extension exists =="
$COMPOSE exec -T "$DB_SERVICE" psql -U "$DB_USER" -d "$DB_NAME" \
  -c "CREATE EXTENSION IF NOT EXISTS postgis;"
echo

# 5) (IMPORTANT) Ensure price_aggregates schema supports duplicated postcodes
#    Best practice: surrogate PK + partial unique index for non-postcode levels
echo "== Migrating/Resetting price_aggregates schema (supports duplicated postcodes) =="
$COMPOSE exec -T "$DB_SERVICE" psql -U "$DB_USER" -d "$DB_NAME" <<'SQL'
BEGIN;

-- Drop dependent view(s) safely
DROP VIEW IF EXISTS price_stats_summary CASCADE;

-- Recreate table cleanly (ETL will repopulate it)
DROP TABLE IF EXISTS price_aggregates CASCADE;

CREATE TABLE price_aggregates (
  id BIGSERIAL PRIMARY KEY,
  level TEXT NOT NULL,
  code  TEXT NOT NULL,
  name  TEXT,
  median_price DOUBLE PRECISION,
  weighted_price DOUBLE PRECISION,
  std_dev DOUBLE PRECISION,
  transaction_count INTEGER,
  lower_bound DOUBLE PRECISION,
  upper_bound DOUBLE PRECISION,
  last_transaction_date TIMESTAMP,
  geom geometry(MULTIPOLYGON, 4326)
);

-- Spatial index
CREATE INDEX price_aggregates_geom_gix
  ON price_aggregates USING GIST (geom);

-- Fast filters and lookups
CREATE INDEX price_aggregates_level_code_idx
  ON price_aggregates(level, code);

-- Extra-fast postcode code lookup (optional)
CREATE INDEX price_aggregates_postcode_code_idx
  ON price_aggregates(code)
  WHERE level='postcode';

-- Enforce uniqueness for everything EXCEPT postcodes
CREATE UNIQUE INDEX price_aggregates_unique_non_postcode
  ON price_aggregates(level, code)
  WHERE level <> 'postcode';

COMMIT;

ANALYZE price_aggregates;
SQL
echo

# 6) Run the full ETL (downloads, loads, aggregates all levels, verifies)
echo "== Running process_data.py inside ETL container =="
$COMPOSE run --rm etl python process_data.py
echo

# 7) Post-run quick sanity checks
echo "== DB sanity checks =="
$COMPOSE exec -T "$DB_SERVICE" psql -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT COUNT(*) AS transactions_rows FROM transactions;"

$COMPOSE exec -T "$DB_SERVICE" psql -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT level, COUNT(*) AS rows FROM price_aggregates GROUP BY level ORDER BY level;"

$COMPOSE exec -T "$DB_SERVICE" psql -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT COUNT(*) AS invalid_geoms FROM price_aggregates WHERE geom IS NULL OR NOT ST_IsValid(geom);"

echo
echo "✅ Full ETL completed."
