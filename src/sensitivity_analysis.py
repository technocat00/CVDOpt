"""
sensitivity_analysis.py
=======================
Feature importance analysis using GP kernel lengthscales,
permutation importance, and partial dependence plots.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. GP Kernel Lengthscale Importance
# ─────────────────────────────────────────────

def lengthscale_importance(gp_model, feature_names: list) -> pd.DataFrame:
    """
    Extract per-feature importance from the fitted GP kernel.

    Physical interpretation:
    - Short lengthscale  → feature changes the output rapidly → HIGH importance
    - Long lengthscale   → feature barely affects output → LOW importance

    Importance score = 1 / lengthscale (normalised to sum to 1).
    """
    kernel = gp_model.model.kernel_

    # Try to extract lengthscale array (anisotropic RBF/Matern)
    ls = None
    for name, param in kernel.get_params().items():
        if name == 'k1__k2__length_scale' or name == 'k2__length_scale':
            ls = np.atleast_1d(param)
            break
        if 'length_scale' in name and not name.endswith('_bounds'):
            val = np.atleast_1d(param)
            if len(val) > 1:
                ls = val
                break

    # Scalar lengthscale fallback — same for all features
    if ls is None or len(ls) == 1:
        # Use perturbation-based method instead
        print("[SensAnalysis] Scalar kernel — using perturbation importance.")
        return perturbation_importance(gp_model, feature_names)

    n_feats = min(len(ls), len(feature_names))
    importance = 1.0 / (ls[:n_feats] + 1e-9)
    importance = importance / importance.sum()

    df = pd.DataFrame({
        'Feature':     feature_names[:n_feats],
        'Lengthscale': np.round(ls[:n_feats], 4),
        'Importance':  np.round(importance, 4),
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\n── GP Kernel Lengthscale Importance ──")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 2. Permutation Feature Importance
# ─────────────────────────────────────────────

def perturbation_importance(gp_model, feature_names: list,
                             X_test: np.ndarray = None,
                             y_test: np.ndarray = None,
                             n_repeats: int = 10,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Permutation importance: shuffle each feature, measure RMSE increase.
    Higher increase → feature is more important.

    If X_test / y_test not provided, generates a synthetic grid.
    """
    rng = np.random.default_rng(random_state)

    if X_test is None or y_test is None:
        # Generate a small evaluation set from the GP itself
        n_eval = 200
        n_feats = len(feature_names)
        X_test = rng.uniform(-2, 2, (n_eval, n_feats))
        y_test, _ = gp_model.predict(X_test, return_std=True)

    y_pred_base, _ = gp_model.predict(X_test, return_std=True)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))

    importances = []
    for i, feat in enumerate(feature_names):
        feat_deltas = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            y_perm, _ = gp_model.predict(X_perm, return_std=True)
            perm_rmse = np.sqrt(mean_squared_error(y_test, y_perm))
            feat_deltas.append(perm_rmse - base_rmse)
        importances.append({
            'Feature':          feat,
            'RMSE_Increase':    round(float(np.mean(feat_deltas)), 5),
            'RMSE_Increase_Std':round(float(np.std(feat_deltas)), 5),
        })

    df = pd.DataFrame(importances)
    df['Importance'] = df['RMSE_Increase'].clip(lower=0)
    total = df['Importance'].sum()
    df['Importance_Norm'] = (df['Importance'] / (total + 1e-9)).round(4)
    df = df.sort_values('RMSE_Increase', ascending=False).reset_index(drop=True)

    print("\n── Permutation Feature Importance ──")
    print(df[['Feature', 'RMSE_Increase', 'RMSE_Increase_Std', 'Importance_Norm']].to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 3. Partial Dependence (1D)
# ─────────────────────────────────────────────

def partial_dependence_1d(gp_model, X_background: np.ndarray,
                           feature_idx: int, feature_name: str,
                           n_grid: int = 50) -> pd.DataFrame:
    """
    Partial dependence of predicted quality on one feature,
    averaging over the background distribution of all other features.
    """
    grid_vals = np.linspace(X_background[:, feature_idx].min(),
                            X_background[:, feature_idx].max(),
                            n_grid)
    pd_means, pd_stds = [], []

    for val in grid_vals:
        X_mod = X_background.copy()
        X_mod[:, feature_idx] = val
        mu, sigma = gp_model.predict(X_mod, return_std=True)
        pd_means.append(float(np.mean(mu)))
        pd_stds.append(float(np.mean(sigma)))

    df = pd.DataFrame({
        feature_name:    grid_vals,
        'mean_quality':  pd_means,
        'mean_std':      pd_stds,
    })
    return df


def partial_dependence_all(gp_model, X_background: np.ndarray,
                            feature_names: list, n_grid: int = 50) -> dict:
    """Compute 1D PDP for all features. Returns dict of DataFrames."""
    pdps = {}
    for i, name in enumerate(feature_names):
        pdps[name] = partial_dependence_1d(gp_model, X_background, i, name, n_grid)
    print(f"[SensAnalysis] Computed partial dependence for {len(feature_names)} features.")
    return pdps


# ─────────────────────────────────────────────
# 4. 2D Interaction Map
# ─────────────────────────────────────────────

def interaction_map_2d(gp_model, X_background: np.ndarray,
                        feat_i: int, feat_j: int,
                        feat_name_i: str, feat_name_j: str,
                        n_grid: int = 20) -> dict:
    """
    2D partial dependence between two features (interaction heatmap).
    Returns meshgrid arrays and predicted quality matrix.
    """
    xi = np.linspace(X_background[:, feat_i].min(), X_background[:, feat_i].max(), n_grid)
    xj = np.linspace(X_background[:, feat_j].min(), X_background[:, feat_j].max(), n_grid)
    Xi, Xj = np.meshgrid(xi, xj)

    Z_mean = np.zeros((n_grid, n_grid))
    Z_std  = np.zeros((n_grid, n_grid))

    for row in range(n_grid):
        for col in range(n_grid):
            X_mod = X_background.copy()
            X_mod[:, feat_i] = Xi[row, col]
            X_mod[:, feat_j] = Xj[row, col]
            mu, sigma = gp_model.predict(X_mod, return_std=True)
            Z_mean[row, col] = float(np.mean(mu))
            Z_std[row, col]  = float(np.mean(sigma))

    return {
        'Xi': Xi, 'Xj': Xj,
        'Z_mean': Z_mean, 'Z_std': Z_std,
        'feat_name_i': feat_name_i,
        'feat_name_j': feat_name_j,
    }


# ─────────────────────────────────────────────
# 5. Summary Report
# ─────────────────────────────────────────────

def sensitivity_report(gp_model, feature_names: list,
                        X_test: np.ndarray = None,
                        y_test: np.ndarray = None) -> pd.DataFrame:
    """
    Unified sensitivity summary: kernel + permutation importances combined.
    """
    perm_df = perturbation_importance(gp_model, feature_names, X_test, y_test)

    try:
        ls_df = lengthscale_importance(gp_model, feature_names)
        merged = perm_df.merge(
            ls_df[['Feature', 'Importance']].rename(columns={'Importance': 'LS_Importance'}),
            on='Feature', how='left'
        )
    except Exception:
        merged = perm_df.copy()
        merged['LS_Importance'] = np.nan

    # Composite rank
    merged['Rank'] = merged['RMSE_Increase'].rank(ascending=False).astype(int)
    print(f"\n{'═'*55}")
    print("  Feature Sensitivity Summary")
    print(f"{'═'*55}")
    print(merged[['Rank', 'Feature', 'RMSE_Increase', 'Importance_Norm']].to_string(index=False))
    return merged


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from sklearn.model_selection import train_test_split
    from data_preprocessing import generate_raw_cvd_data, preprocess_pipeline
    from feature_engineering import add_engineered_features, build_quality_score, get_feature_columns
    from gpr_model import CVDGaussianProcess

    raw_df = generate_raw_cvd_data(120)
    proc_df, _, _ = preprocess_pipeline(raw_df)
    proc_df = add_engineered_features(proc_df)
    proc_df = build_quality_score(proc_df)
    feat_cols = get_feature_columns(proc_df)

    X = proc_df[feat_cols].values
    y = proc_df['quality_score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gp = CVDGaussianProcess(kernel_type='matern').build()
    gp.fit(X_train, y_train)

    report = sensitivity_report(gp, feat_cols, X_test, y_test)
    pdps = partial_dependence_all(gp, X_test, feat_cols[:4])
