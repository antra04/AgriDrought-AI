# 🌱 AgriDrought-AI

### **A County-Level Drought Index Prediction System Using Remote-Sensing Vegetation Signals**

**Developer:** *Antra Tiwari (AI/ML Engineer)*
**Live App:** [https://huggingface.co/spaces/antra04/AgriDrought-AI](https://huggingface.co/spaces/antra04/AgriDrought-AI)


---

## 🚀 Project Overview

**AgriDrought-AI** is an end-to-end geospatial drought forecasting system engineered to predict **weekly agricultural drought indices** across U.S. counties using satellite-derived vegetation signals (NDVI & EVI).

The platform combines:

* High-resolution spatiotemporal preprocessing
* Multi-model machine learning ensembles
* Robust feature engineering pipelines
* Cloud-native deployment via Hugging Face Spaces
* Interactive Gradio-based inference UI

It transforms raw vegetation reflectance values into **operational drought intelligence** for agriculture, climate analytics, and environmental early-warning systems.

---

## 🎯 Why This Project Matters

Accurate drought prediction enables:

* Climate-risk mitigation
* Agricultural decision support
* Water-resource and irrigation planning
* Crop yield protection
* Early-warning systems for extreme climate patterns
* Transparent, reproducible data-driven drought insights

---

## 🛰️ Dataset Summary

### **Spatial & Temporal Coverage**

✔ Nationwide U.S. county-level coverage
✔ Weekly temporal granularity
✔ USDA/USGS standardised **FIPS-based geospatial indexing**

### **Core Training Features**

| Feature                 | Description                            |
| ----------------------- | -------------------------------------- |
| **NDVI**                | Normalized Difference Vegetation Index |
| **EVI**                 | Enhanced Vegetation Index              |
| **County FIPS**         | 5-digit county geocode                 |
| **State FIPS**          | 2-digit state geocode                  |
| **Year / Week / Month** | Temporal encoding                      |
| **Week-of-Year**        | Seasonal behaviour indicator           |
| **sin_week, cos_week**  | Fourier-style seasonal transforms      |
| **ndvi_z, evi_z**       | Standardized vegetation anomalies      |

### **Data Sources**

* Satellite-derived vegetation reflectance datasets
* US county boundary metadata
* Historical environmental time-series

---

## 🧪 Model Architecture

AgriDrought-AI uses a **three-model ensemble** (no meta-stacker for deployment reliability):

### 🌲 **1. Random Forest Regressor**

* Bagging-based tree ensemble
* Insensitive to multicollinearity
* Strong baseline for noisy geospatial signals
* Robust variance control

### ⚡ **2. XGBoost (JSON Booster)**

* High-performance gradient boosting
* Sparse-aware tree splitting
* Uses HF-compatible **native JSON booster** for fast loading
* Captures subtle temporal–vegetation nonlinearities

### 🐈 **3. CatBoost Regressor**

* Implements **ordered boosting** (reduces target leakage)
* Best performance on heterogeneous tabular data
* Handles missing vegetation values gracefully
* Top model by RMSE & R²

---

## 📈 Model Performance

| Model             | RMSE ↓ | R² ↑  | Notes                                      |
| ----------------- | ------ | ----- | ------------------------------------------ |
| **CatBoost**      | ~0.88  | ~0.96 | Best generalization, stable across seasons |
| **XGBoost**       | ~1.00  | ~0.94 | Strong nonlinear modelling                 |
| **Random Forest** | ~1.25  | ~0.91 | Reliable fallback baseline                 |

🏆 **Best Overall:** *CatBoost*
🧱 **Most Stable Contributor:** *Random Forest*

---

## 🔍 System Workflow

### **1. User Inputs**

* NDVI
* EVI
* County FIPS
* State FIPS
* Week Start Date

### **2. Automated Feature Engineering Pipeline**

The system performs dynamic transformations:

* Date parsing → year, month, week
* Seasonal Fourier encodings (sin/cos)
* Vegetation index Z-score normalization
* Deterministic feature ordering to match model training schema
* Data consistency validation

### **3. Multi-Model Inference**

* All three models run independently
* Predictions stored with metadata
* Ranking based on model confidence + validation performance

### **4. Output**

* Final drought index
* Full model comparison table
* Confidence-ranked predictions
* Visualization (Matplotlib bar chart)
* Exportable CSV files

---

## 🌐 Gradio-Based UI (HF Spaces)

### **Single Prediction**

* Input panel
* Instant drought prediction
* Model comparison graph
* Downloadable results

### **Batch CSV Prediction**

* Upload CSV
* Batched inference
* Batch-level graphs
* Processed CSV output

### **Reports Section**

* Loaded model metadata
* HF Hub version details

### **About Section**

* Project documentation embedded in UI

---

## ☁️ Deployment Architecture

### **1. GitHub (Source Code)**

* `app.py`: full Gradio interface + inference pipeline
* `requirements.txt`: deterministic environment spec
* `runtime.txt`: Python version pinning
* `.gitignore`: build artifacts excluded
* Documentation & metadata

### **2. Hugging Face Hub (Model Registry)**

* All models stored under:
  **`antra04/agri-drought-ai-models`**
* Auto-downloaded and cached on first inference
* Versioning supported

### **3. Hugging Face Spaces (App Hosting)**

* Auto-build environment
* Cached model weights
* Session-based scalable endpoint
* Zero-config CI/CD on push

### **4. Pipeline Orchestration**

* Deterministic inference graph
* On-demand model loading
* Stateless HTTP session
* Optimized for low-latency predictions

---

## ⚠️ Limitations

1. **Restricted to U.S. geography** (FIPS-based inference only)
2. **Vegetation anomalies** (wildfire, snow cover, flooding) may affect performance
3. **Not a hydrological drought model** (does not analyze groundwater)
4. **Stacked meta-learner removed** for deployment stability
5. **Trained for in-distribution week ranges** (seasonal generalization only)

---

## 🛠️ Technical Stack (Deep Technical Details)

### **Programming & Core Libraries**

* **Python 3.10** (stable for ML, HF Spaces)
* **NumPy** for vectorized numerical ops
* **Pandas** for high-throughput tabular processing

### **Machine Learning / Modelling**

* **XGBoost** (JSON booster, optimized tree traversal engine)
* **CatBoost** (ordered boosting, symmetric trees)
* **Scikit-Learn** (RF, preprocessing, metrics)

### **Geospatial & Temporal Engineering**

* Custom **Fourier seasonal encoding pipeline**
* Z-score vegetation anomaly modelling
* Automated feature alignment across models
* FIPS-based geospatial indexing

### **Visualization**

* **Matplotlib** for deterministic, reproducible charts

### **Deployment & Cloud Infra**

* **Hugging Face Spaces**

  * Sandboxed Python environment
  * Cached model assets
  * Ephemeral GPU/CPU compute
  * Stateless, scalable service

* **Hugging Face Hub (Model Hosting)**

  * Versioned model artifacts
  * JSON boosters for XGBoost
  * CatBoost binary serialization

### **Front-End / UI**

* **Gradio 4.44**

  * Event-driven inference
  * CSV uploaders
  * Custom theming
  * Session-agnostic layout

### **MLOps + DevOps**

* GitHub-based code versioning
* Automatic app rebuild on push
* Model-hub integration for reproducibility
* Fully containerized HF environment

---

## 🎉 Final Delivery Summary

AgriDrought-AI provides:

✔ Operational drought forecasting pipeline
✔ End-to-end reproducible data → model → inference system
✔ Robust multi-model ensemble
✔ Feature-engineering automation
✔ Cloud-native, zero-maintenance deployment
✔ User-friendly Gradio interface
✔ Fast inference with native boosters
✔ Scalable, transparent model architecture

