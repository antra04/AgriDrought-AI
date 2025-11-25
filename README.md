🌱 AgriDrought-AI
A County-Level Drought Index Prediction System using Remote-Sensing Vegetation Signals

Developer: Antra Tiwari (AI/ML Engineer)
Live App: https://huggingface.co/spaces/antra04/AgriDrought-AI

GitHub Repo: https://github.com/antra04/AgriDrought-AI

🚀 Project Overview

AgriDrought-AI is a geospatial intelligence system engineered to predict weekly drought severity across U.S. counties using satellite-derived vegetation indices.

The solution integrates climate-aware feature engineering, machine-learning ensemble models, and a production-grade Gradio interface deployed on Hugging Face Spaces.

The platform transforms raw NDVI–EVI vegetation signals into actionable drought risk insights for precision agriculture, early-warning systems, crop planning, and climate-impact research.

🎯 Why This Project Matters

Agricultural drought forecasting is essential for:

Reducing climate-risk exposure

Supporting water-resource planning

Optimizing crop decisions

Helping farmers, analysts & policymakers act early

Providing transparent AI-driven drought intelligence

AgriDrought-AI brings together data, models, and deployment into an easy-to-use web tool.

🛰️ Dataset Summary
Region

✔ United States counties (nationwide coverage)
✔ USDA + USGS-aligned FIPS coding system
✔ Weekly granularity

Core Data Fields Used During Training
Feature	Description
NDVI	Normalized Difference Vegetation Index — vegetation greenness
EVI	Enhanced Vegetation Index — improves sensitivity in dense vegetation
County FIPS	Unique 5-digit county geocode
State FIPS	2-digit state identifier
Year	Calendar year
Week	ISO week number
Month	Extracted from week timestamp
Week-of-Year	Used for seasonal modelling
sin_week & cos_week	Seasonal sinusoidal encodings
ndvi_z, evi_z	Z-score normalized vegetation indices
Data Source Type

Historical NDVI/EVI satellite vegetation index records

USA geospatial metadata

Weekly environmental time-series

🧪 Model Architecture

AgriDrought-AI uses a three-model ensemble (stacking removed for simplicity & stability):

🌲 1. Random Forest Regressor

Strong baseline

Handles noisy environmental signals

Excellent variance control

⚡ 2. XGBoost (native JSON booster)

High-performance gradient-boosting system

Best for complex non-linear vegetation patterns

Deployed via fast native booster

🐈 3. CatBoost Regressor

Handles high-dimensional tabular data

Robust to missing values

Captures vegetation trends effectively

📈 Model Performance (Validation Metrics)
Model	RMSE ↓	R² ↑	Notes
CatBoost	~0.88	~0.96	Strongest stability & consistency
XGBoost	~1.00	~0.94	High accuracy, best non-linear detection
Random Forest	~1.25	~0.91	Robust fallback baseline

Best model (overall): CatBoost
Most consistent prediction contribution: Random Forest

🔍 System Workflow
1. Input Features Provided by User

NDVI

EVI

County FIPS

State FIPS

Week Start Date

2. Automated Feature Engineering

The system generates:

year, week, month

sin/cos seasonal encodings

vegetation z-scores

all features in correct model-train order

3. Model Inference

Each model predicts its own drought index

Predictions are ranked by confidence

Results displayed as:

Summary text

Prediction table

Bar chart

Downloadable CSV

4. Output

Drought index (regression score)

Best performing model for that specific input

🌐 UI Overview (Gradio Web App)

The deployed app includes:

🔹 Single Prediction Page

Enter NDVI, EVI, FIPS codes, date

Click “Predict”

See:

Summary

Full model comparison table

Prediction visualization

Downloadable CSV

🔹 Batch CSV Upload

Upload a CSV of multiple counties

Outputs predictions for all rows

Generates batch-level bar graph

🔹 Reports Section

Lists all models loaded from HF Hub

🔹 About Section

Documentation built directly into interface

☁️ Deployment Architecture
1. Code Hosted on GitHub

👉 Repo: https://github.com/antra04/AgriDrought-AI

Includes:

app.py

requirements.txt

runtime.txt

.gitignore

README.md

2. Models Hosted on Hugging Face Hub

All trained models stored under:
🔗 antra04/agri-drought-ai-models

The Gradio app auto-downloads models at runtime.

3. Website Deployed on Hugging Face Spaces

Live app:
🔗 https://huggingface.co/spaces/antra04/AgriDrought-AI

4. Automated Build Pipeline

Hugging Face handles:

Environment creation

Dependency installation

App serving

Model caching

⚠️ Limitations

While AgriDrought-AI is operational and validated, current limitations include:

1. Trained only on U.S. geographic regions

FIPS codes are U.S.-specific

Predictions outside USA are unsupported

2. NDVI/EVI anomalies

Extreme vegetation anomalies (wildfires, snow cover, hurricanes) may skew results

3. Not a hydrological drought model

Predicts vegetation + agricultural drought

Does not model groundwater, reservoir levels, or precipitation

4. Not a full ensemble stacker

Stacking meta-model removed for deployment safety

Only base models are used in final inference

5. Limited temporal extrapolation

Designed for in-distribution week ranges

📦 Installation & Local Usage
1. Clone repo
git clone https://github.com/antra04/AgriDrought-AI
cd AgriDrought-AI

2. Install dependencies
pip install -r requirements.txt

3. Run locally
python app.py


App opens at:
👉 http://127.0.0.1:7860

🛠️ Tech Stack

Python 3.10

Gradio 4.44

XGBoost

CatBoost

scikit-learn

Pandas / NumPy

Matplotlib

Hugging Face Hub + Spaces

GitHub version control

🎉 Final Delivery Summary

AgriDrought-AI delivers:

A fully-automated drought prediction engine

Clean deployment architecture

Reproducible workflow

Rich UI for non-technical users

Fast inference via XGBoost native booster

Scalable & transparent ML pipeline

Easy distribution via HF Spac
