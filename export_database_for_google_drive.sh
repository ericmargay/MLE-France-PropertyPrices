#!/bin/bash

# ================================================================
# EXPORT DATABASE FOR GOOGLE DRIVE DISTRIBUTION
# ================================================================
# This script exports your processed database so you can share it
# Users can then skip the 45-minute ETL process!
# ================================================================

set -e

# Configuration
DB_CONTAINER="france_property_db"
DB_USER="admin"
DB_NAME="property_prices"
EXPORT_FOLDER="france-property-database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================"
echo "EXPORTING DATABASE FOR GOOGLE DRIVE"
echo "================================================================"
echo ""

# ================================================================
# STEP 1: Check if database is running
# ================================================================
echo -e "${BLUE}Step 1/7: Checking database container...${NC}"
if ! docker ps | grep -q $DB_CONTAINER; then
    echo -e "${RED}✗ Database container not running!${NC}"
    echo "Start it with: docker-compose up -d postgres"
    exit 1
fi
echo -e "${GREEN}✓ Database running${NC}"
echo ""

# ================================================================
# STEP 2: Create export folder
# ================================================================
echo -e "${BLUE}Step 2/7: Creating export folder...${NC}"
rm -rf $EXPORT_FOLDER
mkdir -p $EXPORT_FOLDER
echo -e "${GREEN}✓ Folder created: $EXPORT_FOLDER/${NC}"
echo ""

# ================================================================
# STEP 3: Export database
# ================================================================
echo -e "${BLUE}Step 3/7: Exporting database (this takes 1-2 minutes)...${NC}"
docker exec $DB_CONTAINER pg_dump -U $DB_USER -d $DB_NAME \
    --format=custom \
    --compress=9 \
    -f /tmp/property_prices.dump

docker cp $DB_CONTAINER:/tmp/property_prices.dump "$EXPORT_FOLDER/property_prices.dump"

DUMP_SIZE=$(du -h "$EXPORT_FOLDER/property_prices.dump" | cut -f1)
echo -e "${GREEN}✓ Database exported (${DUMP_SIZE})${NC}"
echo ""

# ================================================================
# STEP 4: Create README for users
# ================================================================
echo -e "${BLUE}Step 4/7: Creating README...${NC}"
cat > "$EXPORT_FOLDER/README.txt" << 'EOF'
FRANCE PROPERTY PRICES - PRE-PROCESSED DATABASE
================================================

This package contains the pre-processed PostgreSQL database for the
France Property Prices project.

CONTENTS:
---------
- property_prices.dump    : PostgreSQL database dump (compressed)
- checksums.txt          : MD5 checksums for verification
- README.txt             : This file

DATABASE STATISTICS:
-------------------
- Format: PostgreSQL custom dump (pg_dump format)
- Compression: Level 9
- Tables: transactions, price_aggregates
- Transactions: ~1,000,000+ property sales
- Aggregates: ~437,000+ across 6 hierarchical levels
- Data year: 2023

WHAT'S INCLUDED:
---------------
✓ Country level aggregates (1)
✓ Region level aggregates (13)
✓ Département level aggregates (96)
✓ Commune level aggregates (31,000+)
✓ Postcode level aggregates (6,000+)
✓ Parcel level aggregates (400,000+)

Each aggregate includes:
- Median price per m²
- Weighted price per m²
- Transaction count
- Confidence intervals
- Confidence score
- Last transaction date
- Geographic boundaries (PostGIS geometries)

HOW TO USE:
-----------
1. Clone the repository:
   git clone https://github.com/ericmargay/ML-Test-France-property-prices

2. Run the setup script:
   chmod +x setup_database.sh
   ./setup_database.sh

3. Start the application:
   docker-compose up -d webapp

4. Open in browser:
   http://localhost:8080

TOTAL TIME: ~5 minutes (instead of 45 minutes with ETL!)

REQUIREMENTS:
-------------
- Docker & Docker Compose
- 10GB free disk space
- Mapbox API token (free tier works)

MORE INFO:
----------
GitHub: https://github.com/ericmargay/ML-Test-France-property-prices
Author: Eric Margay (ericmargay@gmail.com)

================================================
EOF

echo -e "${GREEN}✓ README created${NC}"
echo ""

# ================================================================
# STEP 5: Create checksums for verification
# ================================================================
echo -e "${BLUE}Step 5/7: Creating checksums...${NC}"
cd $EXPORT_FOLDER
if command -v md5sum &> /dev/null; then
    md5sum property_prices.dump > checksums.txt
    echo -e "${GREEN}✓ MD5 checksum created${NC}"
elif command -v md5 &> /dev/null; then
    md5 property_prices.dump > checksums.txt
    echo -e "${GREEN}✓ MD5 checksum created${NC}"
else
    echo -e "${YELLOW}⚠ md5sum not available, skipping checksum${NC}"
fi
cd ..
echo ""

# ================================================================
# STEP 6: Create archive
# ================================================================
echo -e "${BLUE}Step 6/7: Creating compressed archive...${NC}"
tar -czf "${EXPORT_FOLDER}.tar.gz" $EXPORT_FOLDER/

ARCHIVE_SIZE=$(du -h "${EXPORT_FOLDER}.tar.gz" | cut -f1)
echo -e "${GREEN}✓ Archive created: ${EXPORT_FOLDER}.tar.gz (${ARCHIVE_SIZE})${NC}"
echo ""

# ================================================================
# STEP 7: Show summary
# ================================================================
echo -e "${BLUE}Step 7/7: Export summary...${NC}"
echo ""
echo "Files in export folder:"
ls -lh $EXPORT_FOLDER/
echo ""
echo "Archive file:"
ls -lh "${EXPORT_FOLDER}.tar.gz"
echo ""

echo "================================================================"
echo -e "${GREEN}✓ EXPORT COMPLETE!${NC}"
echo "================================================================"
echo ""
echo "📦 Archive ready: ${EXPORT_FOLDER}.tar.gz"
echo "📁 Folder: $EXPORT_FOLDER/"
echo "💾 Size: $ARCHIVE_SIZE"
echo ""
echo "NEXT STEPS:"
echo "=========="
echo ""
echo "1️⃣  Upload to Google Drive:"
echo "   - Go to https://drive.google.com"
echo "   - Upload: ${EXPORT_FOLDER}.tar.gz"
echo "   - Right-click → Share"
echo "   - Set to: 'Anyone with the link can view'"
echo "   - Copy the link"
echo ""
echo "2️⃣  Get the FILE_ID from the link:"
echo "   https://drive.google.com/file/d/FILE_ID_HERE/view"
echo "                                  ^^^^^^^^^^^^"
echo ""
echo "3️⃣  Update setup_database.sh:"
echo "   Replace: GOOGLE_DRIVE_FILE_ID=\"PASTE_YOUR_FILE_ID_HERE\""
echo "   With:    GOOGLE_DRIVE_FILE_ID=\"your_actual_file_id\""
echo ""
echo "4️⃣  Test the download:"
echo "   chmod +x setup_database.sh"
echo "   ./setup_database.sh"
echo ""
echo "5️⃣  Commit to Git:"
echo "   git add setup_database.sh README.md"
echo "   git commit -m 'feat: add pre-processed database download'"
echo "   git push origin master"
echo ""
echo "================================================================"
echo ""
echo "🎉 Users can now skip the 45-minute ETL!"
echo "   They just run: ./setup_database.sh"
echo "   And start in 5 minutes!"
echo ""
echo "================================================================"
