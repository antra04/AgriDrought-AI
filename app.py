import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# --- Paths ---
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# --- Optional: use your src helpers if present ---
try:
    from src.predict_stack import predict_stack_df  # uses same training stack logic
    HAVE_SRC_PRED = True
except Exception:
    HAVE_SRC_PRED = False

# --- Load artifacts, prefer stacker assets ---
def _safe_joblib(p):
    try:
        return joblib.load(p)
    except Exception:
        return None

stacker = _safe_joblib(MODELS_DIR / "stacking_meta_ridge.pkl")
stack_scaler = _safe_joblib(MODELS_DIR / "stack_scaler.pkl")
feature_list = _safe_joblib(MODELS_DIR / "feature_list.pkl")

# Fallbacks (optional models for the single-row demo)
cat_model = None
xgb_model = None
rf_model = _safe_joblib(MODELS_DIR / "random_forest.joblib") or _safe_joblib(MODELS_DIR / "random_forest_final.pkl")
lin_model = _safe_joblib(MODELS_DIR / "linear_baseline.pkl")

try:
    # CatBoost optional
    from catboost import CatBoostRegressor
    cb_path = MODELS_DIR / "catboost.cbm"
    if cb_path.exists():
        cat_model = CatBoostRegressor()
        cat_model.load_model(str(cb_path))
except Exception:
    pass

try:
    # XGBoost optional
    import xgboost as xgb
    xgb_path = MODELS_DIR / "xgboost_final.json"
    if xgb_path.exists():
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_path))
except Exception:
    xgb_model = None
    xgb = None

# -------------------
# Inference utilities
# -------------------
def _predict_stack_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preferred path: uses the trained stacker + scaler + feature list.
    Expects df that already contains feature columns used at train time.
    """
    if HAVE_SRC_PRED:
        # your repo's own robust predictor
        return pd.DataFrame({"prediction": predict_stack_df(df, stacker, stack_scaler, feature_list)})
    # light inline implementation if src is not importable on Space
    if stacker is None or stack_scaler is None or feature_list is None:
        raise RuntimeError("Missing models: stacking_meta_ridge.pkl / stack_scaler.pkl / feature_list.pkl")

    # ensure all features exist
    X = df.copy()
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_list]
    X_vals = stack_scaler.transform(X)
    preds = stacker.predict(X_vals)
    return pd.DataFrame({"prediction": preds})

def predict_batch(file_obj):
    try:
        df = pd.read_csv(file_obj.name)
        preds_df = _predict_stack_from_df(df)
        out = pd.concat([df.reset_index(drop=True), preds_df], axis=1)
        # also hand back a CSV bytes for download
        return out, out.to_csv(index=False).encode("utf-8")
    except Exception as e:
        msg = f"Batch inference error: {e}\n{traceback.format_exc()}"
        return pd.DataFrame({"error": [msg]}), b""

def _single_row_predict(ndvi, evi, geoid, statefp, week_start):
    """
    Best-effort single-row. We don’t rebuild rolling/lags here.
    If your feature_list contains engineered columns, they’re zero-filled.
    """
    try:
        wk = pd.to_datetime(week_start)
    except Exception:
        wk = pd.to_datetime(str(week_start).split()[0])

    row = {
        "NDVI": float(ndvi),
        "EVI": float(evi),
        "GEOID": int(geoid) if geoid not in (None, "", "0") else 0,
        "STATEFP": int(statefp) if statefp not in (None, "", "0") else 0,
        "week_start": wk,
    }
    df = pd.DataFrame([row])

    # Try stack first if artifacts are available
    if stacker is not None and stack_scaler is not None and feature_list is not None:
        try:
            pred = _predict_stack_from_df(df)["prediction"].iloc[0]
            return {"stack": float(pred)}
        except Exception:
            pass

    # Fallbacks
    used_cols = feature_list if feature_list is not None else ["NDVI", "EVI"]
    for c in used_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[used_cols].astype(float).values

    out = {}
    if cat_model is not None:
        try:
            out["catboost"] = float(cat_model.predict(X)[0])
        except Exception:
            out["catboost"] = "error"
    if xgb_model is not None and xgb is not None:
        try:
            dmat = xgb.DMatrix(X, feature_names=[str(c) for c in used_cols])
            out["xgboost"] = float(xgb_model.predict(dmat)[0])
        except Exception:
            out["xgboost"] = "error"
    if rf_model is not None:
        try:
            out["random_forest"] = float(rf_model.predict(X)[0])
        except Exception:
            out["random_forest"] = "error"
    if lin_model is not None:
        try:
            out["linear"] = float(lin_model.predict(X)[0])
        except Exception:
            out["linear"] = "error"

    return out or {"error": "No models loaded"}

# -------------
# Gradio UI
# -------------
import gradio as gr

with gr.Blocks(css=".gradio-container {max-width: 1100px !important;}") as demo:
    gr.Markdown("## 🌱 AgriDrought-AI — Drought Stress Predictor")

    with gr.Tab("Batch (recommended)"):
        gr.Markdown(
            "Upload a **feature CSV** (same schema as training). "
            "We’ll run the **stacked model** and give you a downloadable result."
        )
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        df_out = gr.Dataframe(label="Preview + Predictions", wrap=True)
        file_out = gr.File(label="Download predictions.csv")
        btn = gr.Button("Run Inference", variant="primary")
        btn.click(predict_batch, inputs=file_in, outputs=[df_out, file_out])

        gr.Examples(
            examples=[str((ROOT / "examples" / "sample_rows.csv").resolve())],
            inputs=file_in,
            label="Example file"
        )

    with gr.Tab("Quick single-row"):
        with gr.Row():
            ndvi = gr.Slider(0.0, 1.0, value=0.32, step=0.001, label="NDVI")
            evi = gr.Slider(-5.0, 15.0, value=0.20, step=0.001, label="EVI")
        with gr.Row():
            geoid = gr.Textbox(value="19001", label="GEOID (county FIPS)", scale=1)
            statefp = gr.Textbox(value="19", label="STATEFP", scale=1)
            wk = gr.Textbox(value="2023-06-03", label="week_start (YYYY-MM-DD)", scale=1)

        out_json = gr.JSON(label="Predictions")
        btn_row = gr.Button("Predict (single row)")
        btn_row.click(_single_row_predict, inputs=[ndvi, evi, geoid, statefp, wk], outputs=out_json)

if __name__ == "__main__":
    demo.launch()
