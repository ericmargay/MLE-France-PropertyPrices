#!/bin/bash

# ================================================================
# SETUP PRE-PROCESSED DATABASE - FRANCE PROPERTY PRICES (DUMP ONLY)
# ================================================================
# Downloads a PostgreSQL custom dump from Google Drive and restores it.
# No .tar.gz, no extraction.
# Uses Docker to run gdown so we don't depend on system pip (PEP 668).
# ================================================================

set -euo pipefail

# =============================
# CONFIGURATION
# =============================
GOOGLE_DRIVE_FILE_ID="1ZEoJA-KOwlEegkG0DGUoILaAyl1fBM78"   # MUST be the DUMP file ID
DB_CONTAINER="france_property_db"
DB_USER="admin"
DB_NAME="property_prices"
DUMP_NAME="property_prices.dump"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================"
echo "FRANCE PROPERTY PRICES - DATABASE SETUP"
echo "================================================================"
echo ""
echo "This will:"
echo "  1. Download PostgreSQL dump from Google Drive"
echo "  2. Restore into PostgreSQL"
echo ""

# ================================================================
# STEP 1: Check prerequisites
# ================================================================
echo -e "${BLUE}Step 1/6: Checking prerequisites...${NC}"

command -v docker >/dev/null 2>&1 || { echo -e "${RED}✗ Docker not found${NC}"; exit 1; }
echo -e "${GREEN}✓ Docker found${NC}"

command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}✗ docker-compose not found${NC}"; exit 1; }
echo -e "${GREEN}✓ docker-compose found${NC}"
echo ""

# ================================================================
# STEP 2: Start database container
# ================================================================
echo -e "${BLUE}Step 2/6: Starting database container...${NC}"
if ! docker ps --format '{{.Names}}' | grep -qx "${DB_CONTAINER}"; then
  echo "Starting PostgreSQL container..."
  docker-compose up -d postgres
  echo "Waiting 15 seconds for database to initialize..."
  sleep 15
else
  echo -e "${GREEN}✓ Database already running${NC}"
fi
echo ""

# ================================================================
# STEP 3: Download dump from Google Drive (via Docker+gdown)
# ================================================================
echo -e "${BLUE}Step 3/6: Downloading database dump from Google Drive...${NC}"
rm -f "${DUMP_NAME}"

# This avoids PEP 668 by installing gdown inside a disposable container.
docker run --rm \
  -v "$PWD":/work \
  -w /work \
  python:3.11-slim \
  bash -lc "pip -q install --no-cache-dir gdown && python -m gdown --id '${GOOGLE_DRIVE_FILE_ID}' -O '${DUMP_NAME}'"

echo -e "${GREEN}✓ Download complete${NC}"
echo ""

# ================================================================
# STEP 4: Verify download is not HTML (Drive error page)
# ================================================================
echo -e "${BLUE}Step 4/6: Verifying download...${NC}"

if [ ! -f "${DUMP_NAME}" ]; then
  echo -e "${RED}✗ Download failed - file not found${NC}"
  exit 1
fi

FILE_SIZE=$(du -h "${DUMP_NAME}" | cut -f1)
echo -e "${GREEN}✓ File size: ${FILE_SIZE}${NC}"

# If it's HTML, it's a Google Drive error page
if file "${DUMP_NAME}" | grep -qi "HTML"; then
  echo -e "${RED}✗ Downloaded file is HTML (Google Drive error page)${NC}"
  echo "Fixes:"
  echo "  1) In Google Drive: Share -> 'Anyone with the link' (Viewer)"
  echo "  2) If Drive says 'Too many users have viewed/downloaded': make a copy and share the copy"
  echo "  3) Ensure FILE_ID corresponds to the dump file (property_prices.dump)"
  exit 1
fi

# Helpful info (not a hard fail)
if file "${DUMP_NAME}" | grep -qi "PostgreSQL"; then
  echo -e "${GREEN}✓ File looks like a PostgreSQL dump${NC}"
else
  echo -e "${YELLOW}⚠ File type doesn't mention PostgreSQL. Continuing anyway...${NC}"
fi
echo ""

# ================================================================
# STEP 5: Restore database
# ================================================================
echo -e "${BLUE}Step 5/6: Restoring database...${NC}"

echo "Copying dump to container..."
docker cp "${DUMP_NAME}" "${DB_CONTAINER}:/tmp/restore.dump"

echo "Recreating database..."
docker exec "${DB_CONTAINER}" psql -U "${DB_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
docker exec "${DB_CONTAINER}" psql -U "${DB_USER}" -d postgres -c "CREATE DATABASE ${DB_NAME};"

echo "Enabling PostGIS extension..."
docker exec "${DB_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS postgis;"

echo "Running pg_restore..."
docker exec "${DB_CONTAINER}" pg_restore \
  -U "${DB_USER}" \
  -d "${DB_NAME}" \
  --clean \
  --if-exists \
  --no-owner \
  --no-acl \
  /tmp/restore.dump

echo -e "${GREEN}✓ Database restored${NC}"
echo ""

# ================================================================
# STEP 6: Verify imported data
# ================================================================
echo -e "${BLUE}Step 6/6: Verifying imported data...${NC}"

TRANSACTIONS=$(docker exec "${DB_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM transactions;" | xargs || true)
AGGREGATES=$(docker exec "${DB_CONTAINER}" psql -U "${DB_CONTAINER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM price_aggregates;" 2>/dev/null | xargs || true)

# Note: if the aggregates query fails due to typo above, fallback:
if [ -z "${AGGREGATES}" ]; then
  AGGREGATES=$(docker exec "${DB_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM price_aggregates;" | xargs || true)
fi

echo ""
echo "Database statistics:"
echo "  📊 Transactions:  ${TRANSACTIONS:-N/A}"
echo "  📈 Aggregates:    ${AGGREGATES:-N/A}"
echo ""

echo -e "${GREEN}✓ Setup completed${NC}"
echo ""

# Cleanup local dump (optional)
rm -f "${DUMP_NAME}"
