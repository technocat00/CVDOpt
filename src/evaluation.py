"""
evaluation.py
=============
Model evaluation: MAE, RMSE, R², predicted vs actual plots,
and uncertainty calibration analysis for the GPR surrogate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Regression Metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_std: np.ndarray = None) -> dict:
    """
    Comprehensive regression evaluation.

    Metric  | Meaning
    ─────── | ────────────────────────────────────────
    MAE     | Average absolute prediction error
    RMSE    | Penalises larger errors more
    R²      | Fraction of variance explained (1 = perfect)
    MAPE    | Mean absolute percentage error
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    metrics = {
        'MAE':  round(mae, 5),
        'RMSE': round(rmse, 5),
        'R2':   round(r2, 4),
        'MAPE': round(mape, 2),
    }

    if y_std is not None:
        metrics['Mean_Uncertainty'] = round(float(np.mean(y_std)), 5)
        metrics['Std_Uncertainty']  = round(float(np.std(y_std)), 5)

    return metrics


def print_metrics(metrics: dict, label: str = "GPR Model"):
    print(f"\n{'─'*40}")
    print(f"  Evaluation: {label}")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print(f"{'─'*40}")


# ─────────────────────────────────────────────
# 2. Train / Test Split Evaluation
# ─────────────────────────────────────────────

def train_test_evaluation(gp_model, X: np.ndarray, y: np.ndarray,
                           test_size: float = 0.2, random_state: int = 42):
    """
    Split data, fit on train, evaluate on test.
    Returns metrics dict and (y_test, y_pred, y_std) arrays.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    gp_model.fit(X_train, y_train)
    y_pred, y_std = gp_model.predict(X_test, return_std=True)

    metrics = compute_metrics(y_test, y_pred, y_std)
    print_metrics(metrics, label="Train/Test Split")
    return metrics, y_test, y_pred, y_std, X_train, X_test, y_train


# ─────────────────────────────────────────────
# 3. K-Fold Cross-Validation
# ─────────────────────────────────────────────

def kfold_evaluation(gp_class, X: np.ndarray, y: np.ndarray,
                     k: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    K-Fold cross-validation over the full dataset.
    Returns a DataFrame with per-fold and aggregate metrics.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        model = gp_class().build()
        model.fit(X[train_idx], y[train_idx])
        y_pred, y_std = model.predict(X[val_idx], return_std=True)
        m = compute_metrics(y[val_idx], y_pred, y_std)
        m['Fold'] = fold
        fold_results.append(m)

    df = pd.DataFrame(fold_results).set_index('Fold')
    # Aggregate row
    agg = df.agg(['mean', 'std']).round(5)
    print(f"\n{'─'*40}")
    print(f"  {k}-Fold Cross-Validation Results")
    print(f"{'─'*40}")
    print(df.to_string())
    print(f"\n  Aggregate:\n{agg.to_string()}")
    return df, agg


# ─────────────────────────────────────────────
# 4. Uncertainty Calibration
# ─────────────────────────────────────────────

def calibration_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                          y_std: np.ndarray) -> pd.DataFrame:
    """
    Check whether GP confidence intervals are calibrated.

    A well-calibrated GP should contain the true value inside
    its ±z*σ interval with the corresponding probability.

    Expected  | z-score | Interpretation
    ──────────| ────────| ─────────────────────────
    68.3 %    | 1.000   | 1σ interval
    90.0 %    | 1.645   | 90% CI
    95.0 %    | 1.960   | 95% CI
    99.0 %    | 2.576   | 99% CI
    """
    z_table = {
        '68.3%': 1.000,
        '90.0%': 1.645,
        '95.0%': 1.960,
        '99.0%': 2.576,
    }

    errors = np.abs(y_true - y_pred)
    rows = []
    for ci_label, z in z_table.items():
        expected = float(ci_label.rstrip('%')) / 100
        observed = float(np.mean(errors <= z * y_std))
        rows.append({
            'Confidence Level': ci_label,
            'Expected Coverage': expected,
            'Observed Coverage': round(observed, 4),
            'Difference':       round(observed - expected, 4),
            'Calibrated?':      '✓' if abs(observed - expected) < 0.05 else '✗',
        })

    df = pd.DataFrame(rows)
    print(f"\n{'─'*55}")
    print("  Uncertainty Calibration Analysis")
    print(f"{'─'*55}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 5. Learning Curve
# ─────────────────────────────────────────────

def learning_curve(gp_class, X: np.ndarray, y: np.ndarray,
                   train_sizes: list = None, random_state: int = 42) -> pd.DataFrame:
    """
    Evaluate GPR performance as training set size grows.
    Shows how much data is needed for a good surrogate.
    """
    if train_sizes is None:
        n = len(y)
        train_sizes = [int(f * n) for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        train_sizes = [t for t in train_sizes if t >= 5]

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(y))
    X_shuf, y_shuf = X[idx], y[idx]

    # Fixed test set (20 %)
    n_test = max(5, int(0.2 * len(y)))
    X_test, y_test = X_shuf[:n_test], y_shuf[:n_test]
    X_pool, y_pool = X_shuf[n_test:], y_shuf[n_test:]

    rows = []
    for n_train in train_sizes:
        if n_train > len(X_pool):
            break
        model = gp_class().build()
        model.fit(X_pool[:n_train], y_pool[:n_train])
        y_pred, _ = model.predict(X_test, return_std=True)
        m = compute_metrics(y_test, y_pred)
        m['n_train'] = n_train
        rows.append(m)

    df = pd.DataFrame(rows).set_index('n_train')
    print(f"\n{'─'*40}")
    print("  Learning Curve")
    print(f"{'─'*40}")
    print(df[['MAE', 'RMSE', 'R2']].to_string())
    return df


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
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

    gp = CVDGaussianProcess(kernel_type='matern')
    metrics, y_test, y_pred, y_std, *_ = train_test_evaluation(gp, X, y)

    calibration_analysis(y_test, y_pred, y_std)

    cv_df, cv_agg = kfold_evaluation(CVDGaussianProcess, X, y, k=5)
    lc_df = learning_curve(CVDGaussianProcess, X, y)
