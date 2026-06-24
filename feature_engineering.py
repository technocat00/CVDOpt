"""
feature_engineering.py
=======================
Physics-informed derived features and composite quality score
for CVD MoS₂ synthesis optimization.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. Derived / Engineered Features
# ─────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physically meaningful derived features from raw synthesis parameters.

    Feature                 | Physical Rationale
    ------------------------|---------------------------------------------------
    mo_s_ratio              | Controls stoichiometry of MoS₂ (ideal Mo:S ≈ 1:20)
    thermal_exposure        | Temperature × time captures total thermal energy input
    log_pressure            | Pressure spans orders of magnitude → log linearises it
    total_gas_flow          | Affects precursor transport / partial pressures
    temp_per_pressure       | High T + low P → vapour-phase nucleation regime
    growth_rate_proxy       | Rough grain_size / time proxy (if grain_size available)
    """
    df = df.copy()

    # Mo:S precursor ratio (molar proxy via mass)
    # MoCl5 MW ≈ 273.2, S MW ≈ 32.1 → scale factor ≈ 8.5
    if 'mo_conc_mg' in df.columns and 's_conc_mg' in df.columns:
        df['mo_s_ratio'] = df['mo_conc_mg'] / (df['s_conc_mg'] + 1e-9)

    # Thermal exposure (temperature × growth time)
    if 'temperature' in df.columns and 'growth_time_min' in df.columns:
        df['thermal_exposure'] = df['temperature'] * df['growth_time_min']

    # Log pressure
    if 'pressure_torr' in df.columns:
        df['log_pressure'] = np.log1p(df['pressure_torr'])

    # Total gas flow
    if 'carrier_flow_sccm' in df.columns:
        df['total_gas_flow'] = df['carrier_flow_sccm']   # extend with H₂/N₂ if available

    # Temperature-to-pressure ratio (nucleation regime indicator)
    if 'temperature' in df.columns and 'pressure_torr' in df.columns:
        df['temp_per_pressure'] = df['temperature'] / (df['pressure_torr'] + 1e-9)

    # Arrhenius-like term (simplified activation)
    if 'temperature' in df.columns:
        T_K = df['temperature'] + 273.15
        df['arrhenius_term'] = np.exp(-0.5 / (8.314e-3 * T_K))   # dimensionless proxy

    # Replace any inf/-inf with NaN, then fill with column median
    df = df.replace([np.inf, -np.inf], np.nan)
    engineered = ['mo_s_ratio', 'thermal_exposure', 'log_pressure',
                  'total_gas_flow', 'temp_per_pressure', 'arrhenius_term']
    for col in engineered:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[FeatEng] Added engineered features. Total columns: {df.shape[1]}.")
    return df


# ─────────────────────────────────────────────
# 2. Composite Quality Score
# ─────────────────────────────────────────────

QUALITY_WEIGHTS = {
    'grain_size_norm':    0.30,   # Larger grains → fewer grain boundary defects
    'raman_score':        0.25,   # Raman Δk(A1g−E2g) peak separation → layer quality
    'coverage_pct_norm':  0.25,   # Higher coverage → more uniform film
    'pl_intensity_norm':  0.20,   # PL intensity → direct bandgap / optical quality
}


def compute_raman_score(df: pd.DataFrame) -> pd.Series:
    """
    Raman E²₁g − A₁g separation for MoS₂:
    ~18–25 cm⁻¹ for monolayer, shifts with layers / strain.
    Here we score proximity to the ideal monolayer peak position (383 cm⁻¹).
    """
    if 'raman_peak_pos' in df.columns:
        ideal = 383.0
        deviation = np.abs(df['raman_peak_pos'] - ideal)
        max_dev = deviation.max() + 1e-9
        return 1.0 - (deviation / max_dev)   # 1 = perfect, 0 = worst
    return pd.Series(np.zeros(len(df)), index=df.index)


def build_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine normalised quality metrics into a single scalar [0, 1].

    quality_score = 0.30 * grain_size_score
                  + 0.25 * raman_score
                  + 0.25 * coverage_score
                  + 0.20 * pl_score
    """
    df = df.copy()

    # Ensure normalised columns exist
    for raw_col in ['grain_size_nm', 'coverage_pct', 'pl_intensity']:
        norm_col = raw_col + '_norm'
        if norm_col not in df.columns and raw_col in df.columns:
            mn, mx = df[raw_col].min(), df[raw_col].max()
            df[norm_col] = (df[raw_col] - mn) / (mx - mn + 1e-9)

    raman_score = compute_raman_score(df)
    df['raman_score'] = raman_score

    grain_col    = 'grain_size_nm_norm'  if 'grain_size_nm_norm'  in df.columns else 'grain_size_nm'
    coverage_col = 'coverage_pct_norm'   if 'coverage_pct_norm'   in df.columns else 'coverage_pct'
    pl_col       = 'pl_intensity_norm'   if 'pl_intensity_norm'   in df.columns else 'pl_intensity'

    df['quality_score'] = (
        QUALITY_WEIGHTS['grain_size_norm']   * df[grain_col].fillna(0)
      + QUALITY_WEIGHTS['raman_score']       * df['raman_score'].fillna(0)
      + QUALITY_WEIGHTS['coverage_pct_norm'] * df[coverage_col].fillna(0)
      + QUALITY_WEIGHTS['pl_intensity_norm'] * df[pl_col].fillna(0)
    )

    # Clip to [0, 1]
    df['quality_score'] = df['quality_score'].clip(0, 1)

    print(f"[QualScore] quality_score range: "
          f"[{df['quality_score'].min():.3f}, {df['quality_score'].max():.3f}]")
    return df


# ─────────────────────────────────────────────
# 3. Feature Column Selector
# ─────────────────────────────────────────────

TARGET_COLS = ['grain_size_nm', 'raman_peak_pos', 'coverage_pct', 'pl_intensity']
QUALITY_COLS = ['quality_score', 'raman_score',
                'grain_size_nm_norm', 'coverage_pct_norm', 'pl_intensity_norm']

ENGINEERED_FEATURE_NAMES = [
    'mo_s_ratio', 'thermal_exposure', 'log_pressure',
    'total_gas_flow', 'temp_per_pressure', 'arrhenius_term'
]

RAW_FEATURE_NAMES = [
    'temperature', 'pressure_torr', 'growth_time_min',
    'carrier_flow_sccm', 'mo_conc_mg', 's_conc_mg',
    'precursor_type_enc'
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the feature columns to use as GP model inputs."""
    substrate_cols = [c for c in df.columns if c.startswith('sub_')]
    candidates = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES + substrate_cols
    return [c for c in candidates if c in df.columns]


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_preprocessing import generate_raw_cvd_data, preprocess_pipeline

    raw_df = generate_raw_cvd_data(120)
    proc_df, scaler, t_scalers = preprocess_pipeline(raw_df)
    proc_df = add_engineered_features(proc_df)
    proc_df = build_quality_score(proc_df)

    feat_cols = get_feature_columns(proc_df)
    print(f"\nFeature columns ({len(feat_cols)}):\n  {feat_cols}")
    print(f"\nSample quality scores:\n{proc_df['quality_score'].describe()}")
