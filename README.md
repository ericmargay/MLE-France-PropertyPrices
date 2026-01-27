# France Property Prices - Interactive ML Map 🗺️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-green.svg)](https://lightgbm.readthedocs.io/)

**Machine Learning Engineer Challenge**: Geospatial ML-based price estimation and visualization of residential property prices in France.

---

## 🎯 Project Overview

This project uses **Machine Learning** to estimate current property prices (€/m²) from **1,008,568** historical transactions (2023), displayed across **6 hierarchical geographic levels**:

| Level | Count | Description |
|-------|-------|-------------|
| 🇫🇷 **Country** | 1 | Metropolitan France |
| 🗺️ **Region** | 17 | Administrative regions |
| 📍 **Département** | 95 | Departments |
| 🏘️ **Commune** | 31,056 | Municipalities |
| 📮 **Postcode** | 5,848 | Postal zones |
| 🏗️ **Parcel** | 500,000+ | Building footprints (cadastre) |

### Key Features

- ✅ **ML-based price estimation** with LightGBM gradient boosting
- ✅ **Temporal trend adjustment** (extrapolates to 2026)
- ✅ **Hierarchical smoothing** (Bayesian-inspired for sparse data)
- ✅ **Confidence scoring** with prediction intervals
- ✅ **Real building footprints** from French cadastre
- ✅ **Interactive map** with zoom-based level transitions
- ✅ **Two deployment modes**: Database or Static JSON

---

## 🤖 Machine Learning Methodology

### The Challenge

Historical transaction data (2023) needs to estimate **current prices** (2026). Traditional statistical approaches fail because:
- Tree-based models **cannot extrapolate** beyond training data
- Time-weighting with `datetime.now()` produces meaningless weights
- Sparse areas have unreliable estimates

### Our Solution: Three-Component Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PRICE ESTIMATION                               │
│                                                                      │
│   FINAL PRICE = SPATIAL_MODEL × TEMPORAL_ADJUSTMENT                 │
│                 (smoothed with hierarchical prior)                   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   1. SPATIAL MODEL (LightGBM Gradient Boosting)                     │
│      ├─ Learns: WHERE is expensive vs cheap                         │
│      ├─ Features: lat, lon, department, property type, surface      │
│      ├─ Cross-validated: 5-fold CV with R² metric                   │
│      └─ Does NOT extrapolate in time (trees can't!)                 │
│                                                                      │
│   2. TEMPORAL MODEL (Linear Regression)                             │
│      ├─ Learns: HOW prices change over time                         │
│      ├─ Model: log(price) = α + β×year + ε                          │
│      ├─ β = annual trend coefficient (estimated from data)          │
│      └─ Extrapolates from 2023 → 2025                               │
│                                                                      │
│   3. HIERARCHICAL SMOOTHING (Empirical Bayes)                       │
│      ├─ Problem: Sparse zones have high variance                    │
│      ├─ Solution: "Shrink" toward parent region average             │
│      ├─ Weight = min(1, n_transactions / 50)                        │
│      └─ Postcode ← Commune ← Département ← Region ← Country         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **CV R²** | ~0.51 | Explains 51% of price variance |
| **CV MAPE** | ~40% | Mean absolute percentage error |
| **Annual Trend** | Variable | Estimated from data (not assumed) |

### Feature Importance

1. **Latitude** - North/South position (Paris effect)
2. **Log Surface** - Property size
3. **Department** - Regional market effects
4. **Property Type** - Apartment vs House
5. **Longitude** - East/West position

### Confidence Scoring (0-100)

```python
confidence = model_uncertainty (0-40) + data_volume (0-40) + freshness (0-20)

# Model Uncertainty (prediction interval width)
- Narrow interval (<15%): 40 pts
- Wide interval (>70%): 5 pts

# Data Volume (transaction count in zone)
- ≥100 transactions: 40 pts
- <5 transactions: 5 pts

# Freshness (days since last transaction)
- <180 days: 20 pts
- >730 days: 4 pts
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 10GB disk space
- Mapbox access token (free at [mapbox.com](https://mapbox.com))

### 1. Clone & Configure

```bash
git clone https://github.com/ericmargay/MLE-France-Property-Prices
cd MLE-France-Property-Prices

# Create .env file
cat > .env << EOF
POSTGRES_USER=admin
POSTGRES_PASSWORD=changeme
POSTGRES_DB=property_prices
DATABASE_URL=postgresql://admin:changeme@postgres:5432/property_prices
MAPBOX_ACCESS_TOKEN=pk.your_token_here
EOF
```

### 2. Build & Start

```bash
# Build containers
docker-compose build

# Start database
docker-compose up -d postgres
sleep 15  # Wait for PostgreSQL

# Run ML pipeline (includes parcel-level aggregation)
docker-compose run --rm etl python regenerate_with_ml.py

# Start web app
docker-compose up -d webapp

# Open browser
open http://localhost:8080
```

### 3. (Optional) Export for Static Hosting

```bash
# Export all levels to JSON files
docker-compose run --rm etl python export_to_json.py

# Copy to local machine
docker cp $(docker-compose ps -q etl):/app/static/data ./static/data
```

---

## 📁 Project Structure

```
france-property-prices/
├── app/                          # FastAPI web application
│   ├── main.py                   # API endpoints (DB + Static modes)
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/map.js
│   │   └── data/                 # Exported JSON files (static mode)
│   └── templates/
│       ├── index.html
│       └── top-cities.html
│
├── etl/                          # Data processing pipeline
│   ├── process_data.py           # Full ETL pipeline
│   ├── regenerate_with_ml.py     # ML-only regeneration
│   ├── ml_price_model.py         # LightGBM model implementation
│   ├── spatial_aggregation.py    # Geographic aggregation
│   ├── cadastre_downloader.py    # Building footprint downloader
│   └── export_to_json.py         # Export for static hosting
│
├── data/                         # Data storage (gitignored)
│   ├── geometries/               # Administrative boundaries
│   ├── cadastre/                 # Building footprints cache
│   └── ml_price_model.joblib     # Trained model
│
├── docker-compose.yml
└── README.md
```

---

## 🌐 Deployment Options

### Option A: Full Stack (Database)

Best for: Dynamic updates, real-time queries, full parcel support

| Platform | Database | Free Tier | Setup Difficulty |
|----------|----------|-----------|------------------|
| **Railway** | PostgreSQL | $5/month credit | Easy |
| **Render** | PostgreSQL | 90 days free | Easy |
| **Fly.io** | PostgreSQL | 3GB free | Medium |
| **Supabase** | PostgreSQL | 500MB free | Easy |

```bash
# Deploy to Railway
railway login
railway init
railway up
```

### Option B: Static JSON (Recommended for Demo)

Best for: Free hosting, simple deployment, no database management

| Platform | Cost | Setup |
|----------|------|-------|
| **Vercel** | Free | `vercel deploy` |
| **Netlify** | Free | Drag & drop |
| **GitHub Pages** | Free | Push to gh-pages |

```bash
# 1. Export data to JSON
docker-compose run --rm etl python export_to_json.py

# 2. Copy static files
cp -r app/static ./deploy/
cp -r app/templates ./deploy/

# 3. Deploy to Vercel
cd deploy
vercel deploy
```

### File Sizes (Static Mode)

| Level | File | Size | Compressed |
|-------|------|------|------------|
| Country | country.json | 1 KB | - |
| Region | region.json | 50 KB | - |
| Département | departement.json | 200 KB | - |
| Commune | commune.json | 15 MB | 3 MB |
| Postcode | postcode.json | 5 MB | 1 MB |
| Parcel | parcel/*.json | 200 MB | 40 MB |

---

## 🔌 API Reference

### Base URL
```
http://localhost:8080/api
```

### Get Price Aggregates
```http
GET /api/prices/{level}?bbox={minLon},{minLat},{maxLon},{maxLat}
```

**Levels**: `country`, `region`, `departement`, `commune`, `postcode`, `parcel`

**Response**:
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {
      "code": "75",
      "name": "Paris",
      "median_price": 9500.50,
      "lower_bound": 8200.00,
      "upper_bound": 10800.00,
      "confidence_score": 92,
      "estimate_quality": "Very High",
      "transaction_count": 45000
    },
    "geometry": { "type": "MultiPolygon", "coordinates": [...] }
  }]
}
```

### Get Statistics
```http
GET /api/stats
```

### Health Check
```http
GET /health
```

Full docs: http://localhost:8080/docs

---

## 📊 Data Sources

| Source | URL | Description |
|--------|-----|-------------|
| **DVF 2023** | [data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres/) | 1M+ property transactions |
| **Cadastre** | [cadastre.data.gouv.fr](https://cadastre.data.gouv.fr/) | Building footprints |
| **Boundaries** | [france-geojson](https://github.com/gregoiredavid/france-geojson) | Administrative regions |
| **Postcodes** | [OpenDataSoft](https://data.opendatasoft.com/) | Postal code polygons |

---

## 🧪 Verification

```bash
# Check database aggregates
docker exec -it france_property_db psql -U admin -d property_prices -c "
SELECT level, COUNT(*) as zones, 
       ROUND(AVG(median_price)) as avg_price,
       ROUND(AVG(confidence_score)) as avg_conf
FROM price_aggregates 
GROUP BY level 
ORDER BY CASE level
    WHEN 'country' THEN 1 WHEN 'region' THEN 2
    WHEN 'departement' THEN 3 WHEN 'commune' THEN 4
    WHEN 'postcode' THEN 5 WHEN 'parcel' THEN 6
END;"
```

**Expected Output**:
```
   level     |  zones  | avg_price | avg_conf
-------------+---------+-----------+----------
 country     |       1 |     2489  |       57
 region      |      17 |     2508  |       62
 departement |      95 |     2286  |       63
 commune     |  31,056 |     2572  |       56
 postcode    |   5,848 |     2326  |       75
 parcel      | 500,000 |     2500  |       45
```

---

## ✅ Evaluation Criteria

| Criterion | Status | Implementation |
|-----------|--------|----------------|
| ✅ Colored map loading | ✅ | Choropleth with price gradient |
| ✅ Map not laggy | ✅ | Bbox filtering, progressive loading |
| ✅ Refresh on zoom | ✅ | Auto level switching |
| ✅ All 6 levels | ✅ | Country→Region→Dept→Commune→Post→Parcel |
| ✅ Plausible estimates | ✅ | ML model with confidence intervals |
| ✅ Complete data | ✅ | Full 2023 DVF (1M transactions) |
| ✅ Clean code | ✅ | Modular, documented, Docker-based |
| ✅ Robust architecture | ✅ | PostgreSQL + ML + Static export |
| ✅ Top 10 cities | ✅ | `/top-cities` page |
| ✅ **100% ML** | ✅ | LightGBM + Temporal + Hierarchical |
| 🔄 App hosted | Ready | Vercel/Railway deployment ready |

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

## 👤 Author

**Eric Margay** - Machine Learning Engineer

📄 [CV/Resume](https://ericmargay.github.io/DevResumeCV/) | 
📧 [ericmargay@gmail.com](mailto:ericmargay@gmail.com) | 
💼 [LinkedIn](https://linkedin.com/in/ericmargay) | 
🐙 [GitHub](https://github.com/ericmargay)

---

## 📚 References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [DVF Data Documentation](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)
- [PostGIS Documentation](https://postgis.net/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Mapbox GL JS API](https://docs.mapbox.com/mapbox-gl-js/api/)
