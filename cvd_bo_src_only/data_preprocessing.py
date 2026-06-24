"""
data_preprocessing.py
=====================
Handles unit standardization, missing value imputation,
outlier removal, feature scaling, and categorical encoding
for heterogeneous CVD synthesis data mined from literature.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Synthetic Data Generator (simulates literature mining)
# ─────────────────────────────────────────────

def generate_raw_cvd_data(n_samples: int = 120, random_state: int = 42) -> pd.DataFrame:
    """
    Simulate CVD synthesis data as it would appear after mining multiple papers.
    Values are intentionally noisy / mixed-unit to trigger preprocessing steps.
    """
    rng = np.random.default_rng(random_state)

    substrates = rng.choice(['SiO2/Si', 'Sapphire', 'Mica', 'hBN', 'Quartz'], n_samples)
    precursors = rng.choice(['MoCl5+S', 'MoO3+S', 'Mo(CO)6+H2S', 'MoF6+S'], n_samples)

    temperature_C = rng.uniform(650, 900, n_samples)   # some papers use K – fixed in standardize
    pressure_torr = rng.uniform(0.1, 760, n_samples)
    growth_time_min = rng.uniform(5, 120, n_samples)
    carrier_flow_sccm = rng.uniform(10, 500, n_samples)
    mo_conc_mg = rng.uniform(1, 50, n_samples)
    s_conc_mg = rng.uniform(50, 1000, n_samples)

    # ~15 % missing values scattered across quality columns
    def noisy(arr, pct_nan=0.15):
        arr = arr.copy().astype(float)
        mask = rng.random(len(arr)) < pct_nan
        arr[mask] = np.nan
        return arr

    # Quality metrics (target columns)
    grain_size_nm  = noisy(50 + 0.3*temperature_C - 0.02*pressure_torr + rng.normal(0, 10, n_samples))
    raman_peak_pos = noisy(383 + 0.01*temperature_C - 0.005*pressure_torr + rng.normal(0, 1, n_samples))
    coverage_pct   = noisy(np.clip(10 + 0.1*temperature_C - 0.05*pressure_torr + 0.2*growth_time_min + rng.normal(0, 8, n_samples), 0, 100))
    pl_intensity   = noisy(np.abs(0.5 + 0.002*temperature_C - 0.001*pressure_torr + rng.normal(0, 0.3, n_samples)))

    # Inject some unit inconsistencies: 10 % of temperatures stored in Kelvin
    kelvin_mask = rng.random(n_samples) < 0.10
    temperature_C[kelvin_mask] += 273.15          # will be fixed in standardize_units()

    # Inject outliers
    outlier_idx = rng.choice(n_samples, 6, replace=False)
    temperature_C[outlier_idx] += rng.choice([-300, 300], 6)

    df = pd.DataFrame({
        'temperature':       temperature_C,
        'pressure_torr':     pressure_torr,
        'growth_time_min':   growth_time_min,
        'carrier_flow_sccm': carrier_flow_sccm,
        'mo_conc_mg':        mo_conc_mg,
        's_conc_mg':         s_conc_mg,
        'substrate':         substrates,
        'precursor_type':    precursors,
        'grain_size_nm':     grain_size_nm,
        'raman_peak_pos':    raman_peak_pos,
        'coverage_pct':      coverage_pct,
        'pl_intensity':      pl_intensity,
    })

    return df


# ─────────────────────────────────────────────
# 2. Unit Standardization
# ─────────────────────────────────────────────

def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any Kelvin temperatures to Celsius. Flag rows corrected."""
    df = df.copy()
    # Heuristic: valid CVD range is 550–1000 °C → if > 1000, assume Kelvin
    kelvin_mask = df['temperature'] > 1000
    df.loc[kelvin_mask, 'temperature'] -= 273.15
    n_fixed = kelvin_mask.sum()
    if n_fixed:
        print(f"[Standardize] Converted {n_fixed} Kelvin → Celsius entries.")
    return df


# ─────────────────────────────────────────────
# 3. Outlier Removal (IQR-based)
# ─────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, numeric_cols: list, iqr_factor: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where any numeric column falls outside
    [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    df = df.copy()
    initial_len = len(df)
    mask = pd.Series([True] * len(df), index=df.index)

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        mask &= df[col].between(lower, upper)

    df = df[mask].reset_index(drop=True)
    print(f"[Outliers] Removed {initial_len - len(df)} outlier rows. Remaining: {len(df)}.")
    return df


# ─────────────────────────────────────────────
# 4. Missing Value Imputation
# ─────────────────────────────────────────────

def impute_missing(df: pd.DataFrame, numeric_cols: list, strategy: str = 'median') -> pd.DataFrame:
    """Impute numeric NaNs with median (robust) or mean."""
    df = df.copy()
    n_missing_before = df[numeric_cols].isna().sum().sum()
    imp = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imp.fit_transform(df[numeric_cols])
    print(f"[Impute] Filled {n_missing_before} missing numeric values using '{strategy}'.")
    return df


# ─────────────────────────────────────────────
# 5. Categorical Encoding
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substrate → one-hot dummies (affects nucleation → each is distinct).
    Precursor type → label-encoded (ordinal-ish relevance for Mo source).
    """
    df = df.copy()

    # One-hot for substrate
    substrate_dummies = pd.get_dummies(df['substrate'], prefix='sub').astype(int)
    df = pd.concat([df.drop(columns=['substrate']), substrate_dummies], axis=1)

    # Label encode precursor type
    le = LabelEncoder()
    df['precursor_type_enc'] = le.fit_transform(df['precursor_type'])
    df = df.drop(columns=['precursor_type'])

    print(f"[Encode] Substrate one-hot: {list(substrate_dummies.columns)}; "
          f"Precursor label-encoded.")
    return df


# ─────────────────────────────────────────────
# 6. Feature Scaling
# ─────────────────────────────────────────────

def scale_features(df: pd.DataFrame, feature_cols: list, scaler_type: str = 'standard'):
    """
    Standardize features to zero-mean, unit-variance (StandardScaler)
    or [0,1] range (MinMaxScaler).
    Returns (scaled_df, fitted_scaler).
    """
    df = df.copy()
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"[Scale] Applied {scaler_type} scaling to {len(feature_cols)} features.")
    return df, scaler


# ─────────────────────────────────────────────
# 7. Target Normalization
# ─────────────────────────────────────────────

def normalize_targets(df: pd.DataFrame, target_cols: list):
    """Min-max normalize each target to [0, 1] for composite score calculation."""
    df = df.copy()
    scalers = {}
    for col in target_cols:
        mn, mx = df[col].min(), df[col].max()
        df[f'{col}_norm'] = (df[col] - mn) / (mx - mn + 1e-9)
        scalers[col] = (mn, mx)
    print(f"[Normalize] Targets normalized: {target_cols}.")
    return df, scalers


# ─────────────────────────────────────────────
# 8. Master Preprocessing Pipeline
# ─────────────────────────────────────────────

NUMERIC_PROCESS_COLS = [
    'temperature', 'pressure_torr', 'growth_time_min',
    'carrier_flow_sccm', 'mo_conc_mg', 's_conc_mg'
]
TARGET_COLS = ['grain_size_nm', 'raman_peak_pos', 'coverage_pct', 'pl_intensity']


def preprocess_pipeline(df: pd.DataFrame):
    """
    Full preprocessing: standardize → outliers → impute → encode → scale.
    Returns (processed_df, scaler, target_scalers).
    """
    print("=" * 55)
    print("  CVD Data Preprocessing Pipeline")
    print("=" * 55)

    df = standardize_units(df)
    df = remove_outliers(df, NUMERIC_PROCESS_COLS, iqr_factor=3.0)
    df = impute_missing(df, NUMERIC_PROCESS_COLS + TARGET_COLS, strategy='median')
    df = encode_categoricals(df)
    df, target_scalers = normalize_targets(df, TARGET_COLS)

    # Identify all feature columns (post-encoding)
    feature_cols = [c for c in df.columns
                    if c not in TARGET_COLS + [f'{t}_norm' for t in TARGET_COLS]]
    df, scaler = scale_features(df, feature_cols, scaler_type='standard')

    print("=" * 55)
    print(f"  Final dataset shape: {df.shape}")
    print("=" * 55)

    return df, scaler, target_scalers


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    raw_df = generate_raw_cvd_data(n_samples=120)
    raw_df.to_csv('../data/raw_cvd_data.csv', index=False)
    print(f"\nRaw data saved → data/raw_cvd_data.csv  ({raw_df.shape})")

    processed_df, scaler, t_scalers = preprocess_pipeline(raw_df)
    processed_df.to_csv('../data/processed_cvd_data.csv', index=False)
    print(f"Processed data saved → data/processed_cvd_data.csv  ({processed_df.shape})")
