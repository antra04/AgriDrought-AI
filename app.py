
import os
from pathlib import Path
import traceback
import math
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "antra04/agri-drought-ai-models")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HUB_FILES = [
    "catboost.cbm",
    "xgboost_final.json",
    "random_forest_final.pkl",
    "stacking_meta_ridge.pkl",      # still downloaded, but NOT USED
    "scaler.pkl",
    "feature_list.pkl",
    "xgb_features.pkl",
]

# -----------------------
# DEPENDENCIES
# -----------------------
import gradio as gr

_catboost = None
try:
    from catboost import CatBoostRegressor
    _catboost = CatBoostRegressor
except Exception:
    pass

_xgb = None
try:
    import xgboost as xgb
    _xgb = xgb
except Exception:
    pass

from huggingface_hub import hf_hub_download

# -----------------------
# FIXED: Space-safe model downloader
# -----------------------
def ensure_models():
    for fname in HUB_FILES:
        local_path = MODEL_DIR / fname

        if local_path.exists():
            continue

        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=fname,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            print(f"[OK] Downloaded {fname}")
        except Exception as e:
            print(f"[WARN] Could not download {fname}: {e}")

ensure_models()

# -----------------------
# LOADERS
# -----------------------
def safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            return pickle.load(open(path, "rb"))
        except Exception:
            return None

def load_catboost(path: Path):
    if _catboost is None or not path.exists():
        return None
    try:
        m = CatBoostRegressor()
        m.load_model(str(path))
        return m
    except Exception:
        return None

def load_xgb(path: Path):
    if _xgb is None or not path.exists():
        return None
    try:
        booster = _xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception:
        return None

# -----------------------
# LOAD ALL ARTIFACTS
# -----------------------
files = {p.name: p for p in MODEL_DIR.glob("*")}

feature_list = None
if "feature_list.pkl" in files:
    tmp = safe_load(files["feature_list.pkl"])
    if isinstance(tmp, list):
        feature_list = tmp
    elif isinstance(tmp, dict) and "features" in tmp:
        feature_list = tmp["features"]

xgb_features = None
if "xgb_features.pkl" in files:
    tmp = safe_load(files["xgb_features.pkl"])
    if isinstance(tmp, list):
        xgb_features = tmp

models = {
    "catboost": load_catboost(files["catboost.cbm"]) if "catboost.cbm" in files else None,
    "xgboost": load_xgb(files["xgboost_final.json"]) if "xgboost_final.json" in files else None,
    "random_forest_final": safe_load(files["random_forest_final.pkl"]) if "random_forest_final.pkl" in files else None,
    "stacking_meta_ridge": safe_load(files["stacking_meta_ridge.pkl"]) if "stacking_meta_ridge.pkl" in files else None,  # loaded but NOT USED
}

scaler = safe_load(files["scaler.pkl"]) if "scaler.pkl" in files else None

# -----------------------
# DETERMINE FINAL FEATURE COLUMNS
# -----------------------
if feature_list is not None:
    FEATURE_COLS = feature_list
elif xgb_features is not None:
    FEATURE_COLS = xgb_features
else:
    FEATURE_COLS = [
        "ndvi", "evi", "county_fips", "state_fips",
        "year", "week", "month",
        "week_of_year", "sin_week", "cos_week",
        "ndvi_z", "evi_z"
    ]

# -----------------------
# FEATURE ENGINEERING
# -----------------------
def build_features(ndvi, evi, county_fips, state_fips, week_start):
    ndvi = float(ndvi)
    evi = float(evi)
    county_fips = int(county_fips)
    state_fips = int(state_fips)

    try:
        dt = pd.to_datetime(str(week_start))
        year = int(dt.year)
        week = int(dt.isocalendar().week)
        month = int(dt.month)
    except:
        year = 0
        week = 0
        month = 0

    week_of_year = week
    angle = 2 * math.pi * (week_of_year / 52) if week_of_year else 0
    sin_week = math.sin(angle)
    cos_week = math.cos(angle)

    row = {c: 0.0 for c in FEATURE_COLS}

    def setv(keys, val):
        for k in keys:
            if k in row:
                row[k] = val
                return

    setv(["ndvi"], ndvi)
    setv(["evi"], evi)
    setv(["county_fips"], county_fips)
    setv(["state_fips"], state_fips)
    setv(["year"], year)
    setv(["week"], week)
    setv(["month"], month)
    setv(["week_of_year"], week_of_year)
    setv(["sin_week"], sin_week)
    setv(["cos_week"], cos_week)

    try:
        if scaler is not None and hasattr(scaler, "mean_"):
            names = list(scaler.feature_names_in_)
            if "ndvi" in names:
                idx = names.index("ndvi")
                row["ndvi_z"] = (ndvi - scaler.mean_[idx]) / scaler.scale_[idx]
            if "evi" in names:
                idx = names.index("evi")
                row["evi_z"] = (evi - scaler.mean_[idx]) / scaler.scale_[idx]
    except:
        pass

    return pd.DataFrame([row], columns=FEATURE_COLS)

# -----------------------
# PREDICT SINGLE
# -----------------------
def predict_single(county_name, county_fips, state_fips, ndvi, evi, week_start):
    try:
        X = build_features(ndvi, evi, county_fips, state_fips, week_start)

        preds = {}

        if models["catboost"] is not None:
            try:
                preds["catboost"] = float(models["catboost"].predict(X)[0])
            except:
                preds["catboost"] = np.nan

        if models["xgboost"] is not None and _xgb is not None:
            try:
                dmat = _xgb.DMatrix(X.values, feature_names=list(X.columns))
                preds["xgboost"] = float(models["xgboost"].predict(dmat)[0])
            except:
                preds["xgboost"] = np.nan

        # ❌ REMOVED STACKER HERE  
        for k in ["random_forest_final"]:
            m = models.get(k)
            if m is not None:
                try:
                    preds[k] = float(m.predict(X)[0])
                except:
                    preds[k] = np.nan

        df = pd.DataFrame(
            [{"model": m, "prediction": v} for m, v in preds.items()]
        ).sort_values("prediction", ascending=False)

        fig = plt.figure(figsize=(6, 3))
        plt.bar(df["model"], df["prediction"])
        plt.xticks(rotation=25)
        plt.tight_layout()

        csv = df.to_csv(index=False).encode("utf-8")

        summary = f"County={county_name}, NDVI={ndvi}, EVI={evi}, Week={week_start}. Best model → {df.iloc[0].model}"

        return summary, df, fig, ("predictions.csv", csv)

    except Exception as e:
        return f"Error: {e}", pd.DataFrame(), None, None

# -----------------------
# BATCH CSV
# -----------------------
def predict_batch(file):
    if file is None:
        return "Upload CSV.", pd.DataFrame(), None
    try:
        df_in = pd.read_csv(file.name)
    except Exception as e:
        return f"CSV error: {e}", pd.DataFrame(), None

    rows = []
    for _, r in df_in.iterrows():
        _, rep, _, _ = predict_single(
            "", r["county_fips"], r["state_fips"], r["ndvi"], r["evi"], r["week_start"]
        )
        if rep is not None and not rep.empty:
            rep.insert(0, "week_start", r["week_start"])
            rows.append(rep)

    if not rows:
        return "No predictions.", pd.DataFrame(), None

    out = pd.concat(rows, ignore_index=True)
    fig = plt.figure(figsize=(6, 3))
    out.groupby("model")["prediction"].mean().plot(kind="bar")
    plt.tight_layout()

    return f"Batch scored {len(df_in)} rows", out, fig

# -----------------------
# ABOUT PAGE
# -----------------------

ABOUT_MD = """
# 🌱 About AgriDrought-AI

AgriDrought-AI is a **geospatial drought-intelligence system** engineered to predict county-level drought severity using multi-sensor satellite vegetation signals.  
It converts NDVI–EVI time-series and geo-temporal metadata into **actionable drought insights** for agriculture, climate monitoring, crop planning, and risk assessment.

---

## 📌 What the Model Uses

AgriDrought-AI processes a curated weekly dataset based on:

### 🛰️ Remote-sensing vegetation indices  
- **NDVI (Normalized Difference Vegetation Index)** — measures vegetation greenness & crop vigor  
- **EVI (Enhanced Vegetation Index)** — improves sensitivity in dense vegetation and reduces atmospheric noise  

### 🗺️ Spatial identifiers  
- **County FIPS** — unique county-level geocode  
- **State FIPS** — state-level geocode  

### ⏱️ Temporal attributes  
- **Year & Week numbers**  
- **Week-of-year sinusoidal encodings (sin / cos)**  
- **Month**  
- **Derived z-score metrics (ndvi_z, evi_z)**  

This creates a consistent, climate-aware feature space that captures seasonality, phenology, and vegetation stress evolution.

---

## 🤖 Model Architecture

AgriDrought-AI runs a validated ensemble of high-performance regressors:

### 🌲 **Random Forest Regressor**
- Fast, stable baseline  
- Great for noisy environmental data  
- Provides variance reduction & robustness  

### ⚡ **XGBoost (native booster)**  
- High-accuracy gradient boosting  
- Optimized for large-scale tabular/geo-data  
- Deployed via native JSON booster for speed  

### 🐈 **CatBoost Regressor**
- Handles categorical and numeric data smoothly  
- Excellent performance on temporal vegetation patterns  
- Resistant to overfitting and missing-value noise  

### 🧩 Removed: Stacking Meta-Model
The Ridge-based stacker was **deprecated intentionally** to maximize deployment stability on HuggingFace Spaces.  
Only base-model predictions are used for final scoring.

---

## 📊 Performance Snapshot (Validated)

| Model | RMSE ↓ | R² ↑ |
|-------|--------|------|
| **CatBoost** | ~0.88 | ~0.96 |
| **XGBoost**  | ~1.00 | ~0.94 |
| **Random Forest** | ~1.25 | ~0.91 |

These metrics were obtained on held-out county-level weekly samples representing diverse agricultural regions.

---

## 🎯 Why It Matters

AgriDrought-AI empowers early-action decisions for:

- Precision agriculture  
- Crop water-risk assessment  
- Seasonal drought forecasting  
- Resource allocation planning  
- Climate-change impact analysis  
- Academic & applied research  

The system is designed to support scientists, analysts, agronomists, and decision-makers working in climate and agriculture.

---

## 👩‍💻 Developer

**Antra Tiwari**  
AI/ML Engineer 

---
"""

# -----------------------
# UI
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 AgriDrought-AI — Drought Index Predictor")

    with gr.Tab("Single Prediction"):
        with gr.Row():
            county = gr.Textbox(label="County Name", value="Adair")
            cf = gr.Number(label="County FIPS", value=29001)
            sf = gr.Number(label="State FIPS", value=29)

        with gr.Row():
            nd = gr.Number(label="NDVI", value=0.30)
            ev = gr.Number(label="EVI", value=0.20)

        wk = gr.Textbox(label="Week Start (YYYY-MM-DD)", value="2023-06-03")

        btn = gr.Button("🚀 Predict")

        summary = gr.Textbox(label="Summary")
        table = gr.Dataframe()
        plot = gr.Plot()
        dl = gr.File()

        def run_single(a, b, c, d, e, f):
            s, df, fig, csv = predict_single(a, int(b), int(c), d, e, f)
            out_file = None
            if csv:
                name, data = csv
                with open(name, "wb") as fp:
                    fp.write(data)
                out_file = name
            return s, df, fig, out_file

        btn.click(run_single, [county, cf, sf, nd, ev, wk], [summary, table, plot, dl])

    with gr.Tab("Batch CSV"):
        file = gr.File()
        btn2 = gr.Button("Run Batch")
        sum2 = gr.Textbox()
        tab2 = gr.Dataframe()
        plot2 = gr.Plot()
        btn2.click(predict_batch, [file], [sum2, tab2, plot2])

    with gr.Tab("Reports"):
        found = "\n".join(sorted([p.name for p in MODEL_DIR.glob("*")]))
        gr.Markdown("### Artifacts in /models")
        gr.Textbox(value=found, lines=10)

    with gr.Tab("About"):
        gr.Markdown(ABOUT_MD)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
