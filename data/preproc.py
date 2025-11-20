from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import norm


def get_dataset_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    input_csv = script_path.with_name("Final_Statewise_Water_Dataset.csv")
    output_csv = script_path.with_name("Final_Statewise_Water_Dataset_preprocessed_WSI_v3.csv")
    return input_csv, output_csv


def interpolate_and_fill_by_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_interp = ["rainfall", "soil_moisture", "groundwater_level"]
    for col in cols_to_interp:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interpolate within each state to preserve trends
    df[cols_to_interp] = (
        df.groupby("state", group_keys=False)[cols_to_interp]
        .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
    )

    # Fill any remaining gaps with state means
    df[cols_to_interp] = (
        df.groupby("state", group_keys=False)[cols_to_interp].apply(lambda g: g.fillna(g.mean()))
    )
    return df


def add_lpcd_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["population_consumption_per_month"] = pd.to_numeric(
        df["population_consumption_per_month"], errors="coerce"
    )

    denom = df["population"] * 30
    lpcd = df["population_consumption_per_month"] / denom
    lpcd = lpcd.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], pd.NA)
    df["LPCD"] = lpcd
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    for col in cols:
        series = pd.to_numeric(filtered[col], errors="coerce")
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered = filtered[(series >= lower) & (series <= upper)]
    return filtered


def add_zscores(df: pd.DataFrame, cols: list[str], out_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[out_cols] = scaler.fit_transform(df[cols])
    return df


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def compute_entropy_weights(X: pd.DataFrame) -> np.ndarray:
    X_safe = X.fillna(0.0).clip(lower=0.0)
    col_sums = X_safe.sum(axis=0).replace(0, np.nan)
    P = X_safe.divide(col_sums, axis=1)
    n = len(X_safe)
    uniform = np.full((n,), 1.0 / n)
    for j, col in enumerate(X_safe.columns):
        if np.isnan(col_sums[col]):
            P[col] = uniform
    k = 1.0 / np.log(n) if n > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        P_vals = P.values
        logP = np.where(P_vals > 0, np.log(P_vals), 0.0)
        e = -k * np.sum(P_vals * logP, axis=0)
    d = 1.0 - e
    if np.allclose(d.sum(), 0.0) or np.isnan(d.sum()):
        w = np.full((X_safe.shape[1],), 1.0 / X_safe.shape[1])
    else:
        w = d / d.sum()
    return w


def add_wsi_entropy_and_equal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # Scarcity-oriented indicators in [0,1]
    lpcd_s = _minmax(df["LPCD"])             # demand (higher = more stress)
    rainfall_s = 1.0 - _minmax(df["rainfall"])       # higher rainfall = less stress
    soil_s = 1.0 - _minmax(df["soil_moisture"])      # higher soil = less stress
    gw_s = 1.0 - _minmax(df["groundwater_level"])    # higher gw = less stress

    X = pd.DataFrame({
        "LPCD_s": lpcd_s,
        "rainfall_s": rainfall_s,
        "soil_s": soil_s,
        "groundwater_s": gw_s,
    }, index=df.index)

    # Compute entropy weights
    weights = compute_entropy_weights(X)
    weight_map = {col: float(w) for col, w in zip(X.columns, weights)}

    # Entropy-weighted WSI
    df["WSI_entropy"] = X.values.dot(weights)
    df["WSI_entropy_0_100"] = _minmax(df["WSI_entropy"]) * 100.0

    # Equal-weight WSI
    df["WSI_equal"] = X.mean(axis=1)
    df["WSI_equal_0_100"] = _minmax(df["WSI_equal"]) * 100.0

    # PCA-based WSI
    pca = PCA(n_components=1)
    # Fit PCA on the standardized indicators (X is already in [0,1] but PCA often works best on centered data. 
    # However, X here represents "stress" directly. Let's use X directly as it captures the scale of stress.)
    # Actually, standard practice for PCA index is often on standardized variables. 
    # But here X columns are already comparable (0-1 stress indicators).
    # Let's use X directly to find the direction of maximum variance in "stress space".
    pca_comp = pca.fit_transform(X)
    
    # Check if the first component is positively correlated with the mean stress
    # If not, flip it so that higher PCA score means higher stress
    if np.corrcoef(pca_comp.flatten(), X.mean(axis=1))[0, 1] < 0:
        pca_comp = -pca_comp
        
    df["WSI_pca"] = pca_comp.flatten()
    df["WSI_pca_0_100"] = _minmax(df["WSI_pca"]) * 100.0

    # Add weight columns for easy Excel viewing (same value repeated so visible in all rows)
    for k, v in weight_map.items():
        df[f"entropy_weight_{k}"] = v
        
    # Add PCA explained variance for reference (optional, maybe print it)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_[0]:.4f}")

    return df, weight_map


def compute_hybrid_wsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Hybrid WSI using SPEI-style Log-Logistic fitting on a composite surplus score.
    Composite Surplus = Rainfall_s + Soil_s + Groundwater_s - LPCD_s
    (Using the 0-1 scaled versions where higher is better for supply, lower is better for demand)
    Actually, let's stick to the logic:
    Rainfall (high=good), Soil (high=good), GW (high=good), LPCD (high=bad/stress).
    
    We want a "Net Availability" score to fit the distribution to.
    Net Availability = (Rainfall_norm + Soil_norm + GW_norm) - LPCD_norm
    
    Then fit Log-Logistic (Fisk) distribution to this series for each state.
    Transform to CDF -> Z-score (Standard Normal).
    """
    df = df.copy()
    
    # Re-calculate minmax for this specific scope if needed, or use existing if available.
    # The previous function returned df with _s columns but they weren't saved to the main df in the main function variable 'df'.
    # Wait, 'add_wsi_entropy_and_equal' returns 'df' which DOES contain the columns if we assign it back.
    # Let's check 'main': df, weights = add_wsi_entropy_and_equal(df). Yes, df has the new columns.
    # But 'add_wsi_entropy_and_equal' creates X dataframe internally. It doesn't seem to add 'LPCD_s' etc to the returned df.
    # Let's check 'add_wsi_entropy_and_equal' implementation again.
    # It defines lpcd_s etc but puts them in X. It does NOT add them to 'df'.
    # So we need to recalculate them here or modify the other function. 
    # To keep it clean, let's recalculate locally.
    
    lpcd_s = _minmax(df["LPCD"])
    rainfall_s = _minmax(df["rainfall"]) # Note: In entropy function we did 1-minmax for stress. Here we want availability.
    soil_s = _minmax(df["soil_moisture"])
    gw_s = _minmax(df["groundwater_level"])
    
    # Composite Surplus (Higher = More Water Available)
    # We subtract LPCD (Demand)
    # We add Rainfall, Soil, GW (Supply)
    # We shift it to be positive for Log-Logistic fitting (it requires x > 0)
    composite = rainfall_s + soil_s + gw_s - lpcd_s
    
    # Shift to ensure all values are positive (min value + small epsilon)
    # We do this per state or globally? SPEI does it per location (state).
    
    df["WSI_hybrid"] = np.nan
    
    for state in df["state"].unique():
        mask = df["state"] == state
        state_data = composite[mask]
        
        if len(state_data) < 12: # Need enough data points
            continue
            
        # Shift to positive domain for Fisk distribution
        min_val = state_data.min()
        shift_amount = abs(min_val) + 0.01 if min_val <= 0 else 0
        shifted_data = state_data + shift_amount
        
        try:
            # Fit Log-Logistic (Fisk) distribution
            # params: c (shape), loc, scale
            params = stats.fisk.fit(shifted_data)
            
            # Calculate CDF
            cdf = stats.fisk.cdf(shifted_data, *params)
            
            # Handle extreme CDF values to avoid inf in Z-score
            cdf = np.clip(cdf, 0.0001, 0.9999)
            
            # Transform to Z-score (Standard Normal Inverse CDF)
            # SPEI uses this. Positive SPEI = Wet, Negative SPEI = Dry.
            # Our Composite is "Surplus", so High Composite = Wet = Positive Z-score.
            # But WSI is "Stress Index" (High = Bad).
            # So we should probably FLIP the sign. 
            # High Surplus -> High Z -> Low Stress.
            # So WSI_hybrid = -Z.
            z_scores = norm.ppf(cdf)
            
            # Invert so that Higher Value = Higher Stress (Drought)
            wsi_hybrid_val = -z_scores
            
            df.loc[mask, "WSI_hybrid"] = wsi_hybrid_val
            
        except Exception as e:
            print(f"Could not fit Hybrid WSI for {state}: {e}")
            
    # Normalize WSI_hybrid to 0-100 for consistency with other indices
    # Z-scores typically range -3 to +3.
    # We can use minmax on the whole column.
    df["WSI_hybrid_0_100"] = _minmax(df["WSI_hybrid"]) * 100.0
    
    return df


def main() -> None:
    input_csv, output_csv = get_dataset_paths()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load
    df = pd.read_csv(input_csv)

    # 1) Handle missing
    df = interpolate_and_fill_by_state(df)

    # 2) Add LPCD
    df = add_lpcd_column(df)

    # 3) Remove outliers
    cols_for_outliers = ["rainfall", "soil_moisture", "groundwater_level", "LPCD"]
    df = remove_outliers_iqr(df, cols_for_outliers)

    # 4) Add z-scores
    z_in = ["rainfall", "soil_moisture", "groundwater_level", "LPCD"]
    z_out = ["rainfall_z", "soil_moisture_z", "groundwater_z", "LPCD_z"]
    df = add_zscores(df, z_in, z_out)

    # 5) Compute both WSI versions
    df, weights = add_wsi_entropy_and_equal(df)
    
    # 5b) Compute Hybrid WSI
    df = compute_hybrid_wsi(df)

    # 6) Save
    df.to_csv(output_csv, index=False)

    print("Entropy Weights (scarcity indicators):")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved output with both WSI columns and weights to: {output_csv}")


if __name__ == "__main__":
    main()
