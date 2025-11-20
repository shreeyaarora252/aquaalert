import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "gru_model_equal.pth"
SCALER_PATH = BASE_DIR / "model" / "gru_scalers_equal.pkl"
FEATURE_COLS_PATH = BASE_DIR / "model" / "gru_feature_cols_equal.pkl"

DATA_PATH = BASE_DIR / "data" / "gru_predictions_equal.csv"

# -------------------------
# Scarcity logic
# -------------------------
def classify_scarcity(wsi):
    if wsi > 60:
        return "High", "red"
    elif wsi >= 40:
        return "Moderate", "yellow"
    else:
        return "Low", "green"


# -------------------------
# Load artifacts
# -------------------------
def load_artifacts():
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)

    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    from gru_model import GRUForecaster  # uses your existing architecture

    model = GRUForecaster(
        input_size=len(feature_cols) + 1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scalers, feature_cols


# -------------------------
# Prediction using CSV (temporary)

def get_state_predictions():
    """
    Returns ONE prediction per state:
    - Uses nearest future prediction (month_ahead == 1)
    - Converts WSI to scarcity level and color
    """

    df = pd.read_csv(DATA_PATH)

    # Keep only next-month predictions
    if "month_ahead" in df.columns:
        df = df[df["month_ahead"] == 1]

    results = []

    for state, state_df in df.groupby("state"):
        row = state_df.iloc[0]

        # Identify predicted WSI column automatically
        wsi_col = [c for c in df.columns if "WSI" in c][0]
        wsi = float(row[wsi_col])

        # Clip WSI to valid range
        wsi = max(0, min(100, wsi))

        scarcity, color = classify_scarcity(wsi)

        results.append({
            "state": state,
            "predicted_wsi": round(wsi, 2),
            "scarcity": scarcity,
            "color": color
        })

    return results



# -------------------------
# Test locally
# -------------------------
if __name__ == "__main__":
    predictions = get_state_predictions()
    for p in predictions[:5]:
        print(p)
