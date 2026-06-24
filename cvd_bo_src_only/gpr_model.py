"""
gpr_model.py
============
Gaussian Process Regression surrogate model for CVD quality prediction.
Uses Matérn kernel (realistic for physical systems) + WhiteKernel (noise).
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, WhiteKernel, ConstantKernel as C
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Kernel Definitions
# ─────────────────────────────────────────────

def build_kernel(kernel_type: str = 'matern'):
    """
    Kernel          | Use Case
    ─────────────── | ───────────────────────────────────────────
    RBF             | Infinitely smooth response surface
    Matérn (nu=2.5) | More realistic for physical/materials systems
    WhiteKernel     | Captures experimental noise

    Matérn + WhiteKernel is recommended for CVD synthesis data.
    """
    if kernel_type == 'matern':
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )
    elif kernel_type == 'rbf':
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )
    elif kernel_type == 'combined':
        # Mix of Matérn scales for multi-scale physical effects
        kernel = (
            C(0.5) * Matern(length_scale=0.5, nu=1.5)
            + C(0.5) * Matern(length_scale=2.0, nu=2.5)
            + WhiteKernel(noise_level=0.05)
        )
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    return kernel


# ─────────────────────────────────────────────
# 2. GPR Model Wrapper
# ─────────────────────────────────────────────

class CVDGaussianProcess:
    """
    Surrogate model: GPR with Matérn kernel for CVD quality prediction.
    Predicts: (mean, std) for each candidate synthesis condition.
    """

    def __init__(self, kernel_type: str = 'matern', n_restarts: int = 10,
                 normalize_y: bool = True, random_state: int = 42):
        self.kernel_type = kernel_type
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self._is_fitted = False

    def build(self):
        kernel = build_kernel(self.kernel_type)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=self.normalize_y,
            alpha=1e-6,
            random_state=self.random_state
        )
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        if self.model is None:
            self.build()
        self.feature_names = feature_names
        self.model.fit(X, y)
        self._is_fitted = True
        print(f"[GPR] Fitted on {X.shape[0]} samples, {X.shape[1]} features.")
        print(f"[GPR] Kernel: {self.model.kernel_}")
        print(f"[GPR] Log-marginal-likelihood: {self.model.log_marginal_likelihood_value_:.4f}")
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        """Return mean prediction and optionally std (uncertainty)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self.model.predict(X, return_std=return_std)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute evaluation metrics on held-out set."""
        y_pred, y_std = self.predict(X, return_std=True)
        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2   = r2_score(y, y_pred)
        return {
            'MAE':  round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2':   round(r2, 4),
            'mean_uncertainty': round(float(np.mean(y_std)), 4),
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """K-fold cross-validation."""
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        r2_scores, rmse_scores = [], []
        for train_idx, val_idx in kf.split(X):
            m = GaussianProcessRegressor(
                kernel=build_kernel(self.kernel_type),
                n_restarts_optimizer=5,
                normalize_y=self.normalize_y,
                alpha=1e-6
            )
            m.fit(X[train_idx], y[train_idx])
            y_pred = m.predict(X[val_idx])
            r2_scores.append(r2_score(y[val_idx], y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))

        return {
            'cv_r2_mean':   round(np.mean(r2_scores), 4),
            'cv_r2_std':    round(np.std(r2_scores), 4),
            'cv_rmse_mean': round(np.mean(rmse_scores), 4),
            'cv_rmse_std':  round(np.std(rmse_scores), 4),
        }

    def get_lengthscales(self) -> dict:
        """
        Extract per-feature lengthscales from the fitted kernel.
        Shorter lengthscale → parameter has stronger local effect on quality.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        kernel = self.model.kernel_
        ls = {}
        for hp_name, hp_val in kernel.get_params().items():
            if 'length_scale' in hp_name and not hp_name.endswith('_bounds'):
                ls[hp_name] = hp_val
        return ls

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"[GPR] Model saved → {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ─────────────────────────────────────────────
# 3. Uncertainty Calibration Check
# ─────────────────────────────────────────────

def calibration_check(model: CVDGaussianProcess, X_test: np.ndarray,
                      y_test: np.ndarray, confidence_levels: list = None):
    """
    For a well-calibrated GP: the fraction of test points inside the
    predicted ±z*std interval should match the expected coverage.
    """
    if confidence_levels is None:
        confidence_levels = [0.68, 0.90, 0.95, 0.99]

    y_pred, y_std = model.predict(X_test, return_std=True)
    errors = np.abs(y_test - y_pred)

    z_map = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    results = {}
    for cl in confidence_levels:
        z = z_map.get(cl, 1.96)
        coverage = np.mean(errors <= z * y_std)
        results[f'{int(cl*100)}% CI'] = {
            'expected': cl,
            'observed': round(float(coverage), 3),
            'calibrated': abs(coverage - cl) < 0.05
        }
    return results


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from sklearn.model_selection import train_test_split
    from data_preprocessing import generate_raw_cvd_data, preprocess_pipeline
    from feature_engineering import add_engineered_features, build_quality_score, get_feature_columns

    raw_df = generate_raw_cvd_data(120)
    proc_df, scaler, _ = preprocess_pipeline(raw_df)
    proc_df = add_engineered_features(proc_df)
    proc_df = build_quality_score(proc_df)

    feat_cols = get_feature_columns(proc_df)
    X = proc_df[feat_cols].values
    y = proc_df['quality_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gp = CVDGaussianProcess(kernel_type='matern').build()
    gp.fit(X_train, y_train, feature_names=feat_cols)

    metrics = gp.score(X_test, y_test)
    print(f"\nTest metrics: {metrics}")

    cv_metrics = gp.cross_validate(X, y, cv=5)
    print(f"CV metrics:   {cv_metrics}")

    calib = calibration_check(gp, X_test, y_test)
    print(f"\nCalibration:\n{pd.DataFrame(calib).T}")
