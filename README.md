# 🌱 AgriDrought-AI

### A County-Level Drought Index Prediction System Using Remote-Sensing Vegetation Signals

**Developer:** *Antra Tiwari (AI/ML Engineer)* • **Live App:** [https://huggingface.co/spaces/antra04/AgriDrought-AI](https://huggingface.co/spaces/antra04/AgriDrought-AI) • **GitHub:** [https://github.com/antra04/AgriDrought-AI](https://github.com/antra04/AgriDrought-AI)

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Project Matters](#why-this-project-matters)
3. [Dataset Summary](#dataset-summary)
4. [Model Architecture](#model-architecture)
5. [System Workflow](#system-workflow)
6. [Application UI](#application-ui)
7. [Deployment Architecture](#deployment-architecture)
8. [Technical Stack](#technical-stack)
9. [Limitations](#limitations)
10. [Final Delivery Summary](#final-delivery-summary)

---

## 🚀 Project Overview

AgriDrought-AI is an end-to-end drought forecasting system engineered to generate weekly county-level agricultural drought indices across the United States using satellite-derived vegetation reflectance indicators (NDVI and EVI). The platform combines high-resolution seasonal encoding, multi-model machine learning inference, automated feature pipelines, and cloud-native deployment on Hugging Face Spaces. It transforms raw vegetation signals into reliable, research-grade drought intelligence for agriculture, climate research, disaster preparedness, and environmental analytics.

---

## 🎯 Why This Project Matters

Agricultural droughts impact crop yield, food security, water allocation, and long-term climate resilience. AgriDrought-AI delivers an interpretable, data-driven, and fast drought prediction mechanism that supports:

* Early warning and climate-risk mitigation
* Precision agriculture and crop planning
* Water-resource and irrigation scheduling
* Research-grade environmental monitoring
* Transparent and reproducible AI-driven forecasting

---

## 🛰️ Dataset Summary

### Spatial & Temporal Characteristics

* Nationwide coverage of all U.S. counties
* Weekly temporal resolution
* USDA/USGS-compliant geospatial keys (FIPS codes)

### Training Features

| Feature             | Description                            |
| ------------------- | -------------------------------------- |
| NDVI                | Normalized Difference Vegetation Index |
| EVI                 | Enhanced Vegetation Index              |
| County FIPS         | 5-digit geocode identifier             |
| State FIPS          | 2-digit geocode identifier             |
| Year / Week / Month | Parsed temporal features               |
| Week-of-Year        | Seasonal periodic behavior             |
| sin_week / cos_week | Fourier seasonal encoding              |
| ndvi_z / evi_z      | Vegetation anomaly z-scores            |

### Data Sources

* Historical MODIS-like vegetation datasets
* US county boundary metadata
* Weekly environmental time-series

---

## 🧪 Model Architecture

AgriDrought-AI employs a robust multi-model ensemble architecture optimized for temporal vegetation patterns.

### Ensemble Components

#### **🌲 Random Forest Regressor**

* Bagging-based tree ensemble
* High stability against noisy remote-sensing signals
* Strong variance minimization

#### **⚡ XGBoost (Native JSON Booster)**

* Gradient-boosted decision trees with sparse-aware splitting
* HF-compatible lightweight JSON booster
* Efficient runtime inference

#### **🐈 CatBoost Regressor**

* Ordered boosting to reduce overfitting
* Highly effective on heterogeneous tabular vegetation data
* Best validation performance (RMSE ~0.88, R² ~0.96)

### Validation Performance

| Model         | RMSE ↓ | R² ↑  | Notes                           |
| ------------- | ------ | ----- | ------------------------------- |
| CatBoost      | ~0.88  | ~0.96 | Best generalization & stability |
| XGBoost       | ~1.00  | ~0.94 | Strong non-linear modeling      |
| Random Forest | ~1.25  | ~0.91 | Reliable fallback baseline      |

---

## 🔍 System Workflow

AgriDrought-AI performs automated ML inference using a structured pipeline.

### **High-Level Workflow**

1. User inputs NDVI, EVI, FIPS codes, and date
2. System performs deterministic feature engineering
3. All three models perform independent inference
4. Predictions are aggregated and ranked
5. Outputs include tables, summaries, graphs, and downloadable data

---

## 🧩 Architecture Pipeline (Technical Diagram)

```
                          ┌────────────────────────┐
                          │      User Inputs       │
                          │ NDVI, EVI, FIPS, Date  │
                          └─────────────┬──────────┘
                                        │
                             Feature Engineering
                                        │
          ┌────────────────────────────────────────────────────────┐
          │ - Temporal parsing (year, month, ISO week)             │
          │ - Seasonal encoding (sin/cos Fourier transforms)       │
          │ - Z-score vegetation anomalies (ndvi_z, evi_z)         │
          │ - Feature ordering & schema validation                 │
          └────────────────────────────────────────────────────────┘
                                        │
                                 Model Inference
                                        │
     ┌──────────────────┬─────────────────────┬──────────────────┐
     │ Random Forest    │     XGBoost         │     CatBoost     │
     │ (Bagging)        │ (JSON booster)      │ (Ordered Boost)  │
     └─────────┬────────┴─────────┬──────────┴─────────┬────────┘
               │                  │                     │
               └─────────── Prediction Aggregator ─────┘
                                │
                      ┌───────────────────────┐
                      │ Final Output Layer     │
                      │ - Summary insights     │
                      │ - Model comparison     │
                      │ - Graph generation     │
                      │ - CSV export           │
                      └───────────────────────┘
```

---

## 🌐 Application UI

### Single Prediction Interface

* Input-driven vegetation + geospatial parameters
* Real-time drought index prediction
* Model confidence ranking
* Interactive visualization

### Batch CSV Inference

* Supports multi-county weekly predictions
* Batch-level comparison plots
* Exportable processed datasets

### Reports Section

* Loaded model metadata
* Hugging Face Hub model references

### About Section

* Embedded project documentation

---

## ☁️ Deployment Architecture

### Hugging Face Spaces (App Hosting)

* Stateless containerized environment
* Auto-build on code push
* Cached model weights for low-latency inference
* Session-level scaling

### Hugging Face Hub (Model Registry)

* Models stored under: `antra04/agri-drought-ai-models`
* JSON booster storage for XGBoost
* Binary CatBoost serialization
* Versioned model artifacts for reproducibility

### GitHub (Source Management)

Contains:

* `app.py` (Gradio UI + inference logic)
* `requirements.txt` (dependency locking)
* `runtime.txt` (Python version pinning)
* `assets/` (graphs, documentation, resources)

### Orchestration Mechanics

* Deterministic feature mapping
* On-demand model loading
* Lightweight inference graph
* Zero-maintenance deployment

---

## 🛠️ Technical Stack (Professional Detail)

### Languages & Core Libraries

* **Python 3.10** with deterministic environment constraints
* **NumPy/Pandas** for optimized tabular operations & vector math

### Machine Learning Frameworks

* **CatBoost** – symmetric tree structures, oblivious splits
* **XGBoost** – histogram-based boosting, JSON boosters for deployment
* **Scikit-learn** – RandomForest, preprocessing utilities

### Feature Engineering Framework

* Fourier-based seasonality transformer
* Custom FIPS geospatial encoder
* Vegetation anomaly normalization (Z-score pipeline)
* Consistent schema alignment across models

### Visualization Layer

* Matplotlib (static, reproducible plot generation)

### Cloud & Deployment

* Hugging Face Spaces (containerized serving)
* Hugging Face Hub (artifact registry)
* GitHub (CI versioning, rollout automation)

### MLOps & Versioning

* Model version tracing
* Automatic server rebuild upon push
* Fully deterministic inference environment

---

## ⚠️ Limitations

* Model performance limited to U.S. geography (FIPS-dependent)
* Extreme vegetation anomalies (wildfire, heavy snow, hurricanes) may reduce accuracy
* Not designed for hydrological or groundwater droughts
* No meta-stacking layer (removed for stability)
* Optimized for in-distribution weekly ranges

---

## 🎉 Final Delivery Summary

AgriDrought-AI delivers a production-ready, interpretable, and scalable geospatial drought prediction system. It provides:

* End-to-end ML pipeline with robust preprocessing
* Multi-model ensemble forecasting
* Automated seasonal and vegetation feature engineering
* Cloud-hosted, zero-maintenance Gradio application
* Transparent, reproducible drought analytics
* Industry-grade architecture suitable for research & deployment

