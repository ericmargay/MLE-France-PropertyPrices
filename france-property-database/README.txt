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
