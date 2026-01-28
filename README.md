# France Property Prices - ML Price Estimation 🗺️

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-blue?style=for-the-badge)](https://ericmargay.github.io/MLE-France-PropertyPrices/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-green.svg)](https://lightgbm.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)

> **Machine Learning Engineer Challenge**: Geospatial ML-based price estimation and visualization of residential property prices across France.

## 🔗 Live Demo

**[👉 https://ericmargay.github.io/MLE-France-PropertyPrices/](https://ericmargay.github.io/MLE-France-PropertyPrices/)**

![France Property Prices Map](docs/data/image.png)

---

## 📊 Project Summary

This project uses **Machine Learning** to estimate current property prices (€/m²) from **~1 million** historical transactions (2023), displayed across **6 hierarchical geographic levels**:

| Level | Count | Description |
|-------|-------|-------------|
| 🇫🇷 Country | 1 | Metropolitan France |
| 🗺️ Region | 17 | Administrative regions |
| 📍 Département | 95 | Departments |
| 🏘️ Commune | 31,000+ | Municipalities |
| 📮 Postcode | 5,800+ | Postal zones |
| 🏗️ Parcel | 500,000+ | Building footprints from cadastre |

---

## 🤖 ML Methodology

### The Challenge

Historical transaction data (2023) needs to estimate **current prices** (January 2026). Traditional approaches fail because:
- Tree-based models **cannot extrapolate** beyond training data
- Sparse areas have unreliable estimates due to limited data

### Solution: Three-Component Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PRICE ESTIMATION                          │
│                                                                 │
│   FINAL PRICE = SPATIAL_MODEL × TEMPORAL_ADJUSTMENT             │
│                 (smoothed with hierarchical prior)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. SPATIAL MODEL (LightGBM Gradient Boosting)                │
│      • Learns spatial price patterns (WHERE is expensive)      │
│      • Features: lat, lon, department, property type, surface  │
│      • Cross-validated R² ≈ 0.51                               │
│                                                                 │
│   2. TEMPORAL MODEL (Linear Regression)                        │
│      • Learns price trends over time                           │
│      • log(price) = α + β×year + ε                             │
│      • Extrapolates from 2023 → 2026                           │
│                                                                 │
│   3. HIERARCHICAL SMOOTHING (Empirical Bayes)                  │
│      • Sparse zones "shrink" toward parent region average      │
│      • Postcode ← Commune ← Département ← Region ← Country     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Confidence Scoring (0-100)

Each estimate includes a confidence score based on:
- **Model uncertainty** (prediction interval width)
- **Data volume** (transaction count in zone)
- **Data freshness** (recency of transactions)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | LightGBM, scikit-learn |
| Database | PostgreSQL + PostGIS |
| Backend | FastAPI, Python 3.11 |
| Frontend | Mapbox GL JS, Vanilla JS |
| ETL | Pandas, GeoPandas |
| Deployment | Docker, GitHub Pages |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ericmargay/MLE-France-PropertyPrices
cd MLE-France-PropertyPrices

# Create environment file
cat > .env << EOF
POSTGRES_USER=admin
POSTGRES_PASSWORD=changeme
POSTGRES_DB=property_prices
DATABASE_URL=postgresql://admin:changeme@postgres:5432/property_prices
MAPBOX_ACCESS_TOKEN=your_token_here
EOF

# Build and run
docker-compose build
docker-compose up -d postgres
docker-compose run --rm etl python regenerate_with_ml.py
docker-compose up -d webapp

# Open http://localhost:8080
```

---

## 📁 Project Structure

```
MLE-France-PropertyPrices/
├── app/                          # FastAPI web application
│   ├── main.py                   # API endpoints
│   └── templates/                # HTML templates
│
├── etl/                          # Data processing pipeline
│   ├── ml_price_model.py         # LightGBM model
│   ├── regenerate_with_ml.py     # ML aggregation pipeline
│   ├── spatial_aggregation.py    # Geographic processing
│   └── export_to_json.py         # Static export
│
├── docs/                         # GitHub Pages (static demo)
│   ├── index.html
│   └── data/*.json
│
└── docker-compose.yml
```

---

## 📊 Data Sources

| Source | Description |
|--------|-------------|
| [DVF 2023](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres/) | ~1M property transactions |
| [Cadastre](https://cadastre.data.gouv.fr/) | Building footprints |
| [Admin Boundaries](https://github.com/gregoiredavid/france-geojson) | Regions, departments, communes |

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Data | 997,252 transactions |
| CV R² Score | 0.51 |
| Geographic Zones | 537,000+ |
| Prediction Target | January 2026 |

### Top Feature Importance
1. Latitude (North/South - Paris effect)
2. Property Surface
3. Department
4. Property Type (Apartment vs House)
5. Longitude

---

## 👤 Author

**Eric Margay** - Machine Learning Engineer

[📄 Resume](https://ericmargay.github.io/DevResumeCV/) • 
[📧 Email](mailto:ericmargay@gmail.com) • 
[💼 LinkedIn](https://linkedin.com/in/ericmargay) • 
[🐙 GitHub](https://github.com/ericmargay)

---

## 📝 License

MIT License
