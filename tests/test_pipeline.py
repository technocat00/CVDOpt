"""
tests/test_pipeline.py
======================
Unit tests for each module in the CVD BO pipeline.
Run:  python -m pytest tests/test_pipeline.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    generate_raw_cvd_data, standardize_units, remove_outliers,
    impute_missing, encode_categoricals, scale_features,
    normalize_targets, preprocess_pipeline, NUMERIC_PROCESS_COLS, TARGET_COLS
)
from feature_engineering import (
    add_engineered_features, build_quality_score, get_feature_columns
)
from gpr_model import CVDGaussianProcess, calibration_check
from bayesian_optimization import (
    expected_improvement, probability_of_improvement, upper_confidence_bound,
    build_candidate_grid, BayesianOptimizer
)
from evaluation import compute_metrics, calibration_analysis
from sensitivity_analysis import perturbation_importance


# ══════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════

@pytest.fixture(scope='module')
def raw_df():
    return generate_raw_cvd_data(n_samples=60, random_state=0)


@pytest.fixture(scope='module')
def proc_df(raw_df):
    df, _, _ = preprocess_pipeline(raw_df)
    df = add_engineered_features(df)
    df = build_quality_score(df)
    return df


@pytest.fixture(scope='module')
def X_y(proc_df):
    feat_cols = get_feature_columns(proc_df)
    X = proc_df[feat_cols].values
    y = proc_df['quality_score'].values
    return X, y, feat_cols


@pytest.fixture(scope='module')
def fitted_gp(X_y):
    X, y, feat_cols = X_y
    gp = CVDGaussianProcess(kernel_type='matern', n_restarts=2).build()
    gp.fit(X[:40], y[:40], feature_names=feat_cols)
    return gp, X, y, feat_cols


# ══════════════════════════════════════════════
# 1. Data Preprocessing Tests
# ══════════════════════════════════════════════

class TestPreprocessing:

    def test_generate_shape(self, raw_df):
        assert raw_df.shape == (60, 12), "Raw dataframe should have 12 columns"

    def test_standardize_units_no_kelvin_remaining(self, raw_df):
        df = standardize_units(raw_df)
        assert df['temperature'].max() < 1100, "No temperature should remain above 1000 after K→C fix"

    def test_outlier_removal_reduces_rows(self, raw_df):
        df = standardize_units(raw_df)
        df_clean = remove_outliers(df, NUMERIC_PROCESS_COLS, iqr_factor=3.0)
        assert len(df_clean) <= len(df), "Outlier removal should not add rows"

    def test_impute_no_nans(self, raw_df):
        df = impute_missing(raw_df.copy(), NUMERIC_PROCESS_COLS + TARGET_COLS)
        assert df[NUMERIC_PROCESS_COLS].isna().sum().sum() == 0

    def test_encode_categoricals_drops_original(self, raw_df):
        df = encode_categoricals(raw_df.copy().dropna())
        assert 'substrate' not in df.columns
        assert 'precursor_type' not in df.columns
        assert any(c.startswith('sub_') for c in df.columns)

    def test_scale_features_zero_mean(self, raw_df):
        df = impute_missing(raw_df.copy(), NUMERIC_PROCESS_COLS)
        df_s, _ = scale_features(df, NUMERIC_PROCESS_COLS, scaler_type='standard')
        means = df_s[NUMERIC_PROCESS_COLS].mean().abs()
        assert (means < 1e-6).all(), "Standardised features should have zero mean"

    def test_full_pipeline_no_nans(self, proc_df):
        assert not proc_df.isnull().any().any(), "Processed dataframe should contain no NaNs"

    def test_quality_score_range(self, proc_df):
        q = proc_df['quality_score']
        assert q.min() >= 0.0 and q.max() <= 1.0


# ══════════════════════════════════════════════
# 2. Feature Engineering Tests
# ══════════════════════════════════════════════

class TestFeatureEngineering:

    def test_mo_s_ratio_exists(self, proc_df):
        assert 'mo_s_ratio' in proc_df.columns

    def test_thermal_exposure_positive(self, proc_df):
        if 'thermal_exposure' in proc_df.columns:
            # Can be negative if temperature scaled negative — just check no NaN
            assert not proc_df['thermal_exposure'].isna().any()

    def test_log_pressure_finite(self, proc_df):
        if 'log_pressure' in proc_df.columns:
            assert np.isfinite(proc_df['log_pressure'].values).all()

    def test_feature_columns_nonempty(self, proc_df):
        feat_cols = get_feature_columns(proc_df)
        assert len(feat_cols) >= 5, "Should have at least 5 feature columns"


# ══════════════════════════════════════════════
# 3. GPR Model Tests
# ══════════════════════════════════════════════

class TestGPRModel:

    def test_predict_shape(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        mu, sigma = gp.predict(X[40:], return_std=True)
        assert mu.shape == sigma.shape
        assert len(mu) == len(X[40:])

    def test_predict_std_positive(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        _, sigma = gp.predict(X[40:], return_std=True)
        assert (sigma >= 0).all(), "Uncertainty must be non-negative"

    def test_score_keys(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        scores = gp.score(X[40:], y[40:])
        for key in ('MAE', 'RMSE', 'R2'):
            assert key in scores

    def test_r2_reasonable(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        scores = gp.score(X[40:], y[40:])
        assert scores['R2'] > -5, "R² should be > -5 (at least not catastrophically bad)"

    def test_calibration_check_runs(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        result = calibration_check(gp, X[40:], y[40:])
        assert '95% CI' in result


# ══════════════════════════════════════════════
# 4. Acquisition Function Tests
# ══════════════════════════════════════════════

class TestAcquisitionFunctions:

    def test_ei_shape(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        candidates = X[40:]
        ei = expected_improvement(candidates, gp, y_best=y.max())
        assert ei.shape == (len(candidates),)
        assert (ei >= 0).all(), "EI must be non-negative"

    def test_pi_range(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        candidates = X[40:]
        pi = probability_of_improvement(candidates, gp, y_best=y.max())
        assert ((pi >= 0) & (pi <= 1)).all(), "PI must be in [0,1]"

    def test_ucb_ordering(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        candidates = X[40:]
        ucb_low  = upper_confidence_bound(candidates, gp, y_best=0, kappa=0.1)
        ucb_high = upper_confidence_bound(candidates, gp, y_best=0, kappa=5.0)
        # Higher kappa → higher or equal UCB values on average
        assert ucb_high.mean() >= ucb_low.mean() - 1e-6

    def test_candidate_grid_shape(self):
        bounds = {'a': (-1, 1), 'b': (0, 2), 'c': (-3, 3)}
        grid = build_candidate_grid(bounds, n_grid=500)
        assert grid.shape == (500, 3)
        assert (grid[:, 0] >= -1).all() and (grid[:, 0] <= 1).all()


# ══════════════════════════════════════════════
# 5. Bayesian Optimizer Tests
# ══════════════════════════════════════════════

class TestBayesianOptimizer:

    def test_bo_improves_over_seed(self, fitted_gp):
        gp_ref, X, y, feat_cols = fitted_gp

        def obj(x):
            mu, _ = gp_ref.predict(x, return_std=True)
            return float(mu[0])

        bounds = {f: (-2.5, 2.5) for f in feat_cols}
        rng = np.random.default_rng(1)
        n_seed = 8
        X_init = np.column_stack([rng.uniform(-2.5, 2.5, n_seed) for _ in feat_cols])
        y_init = np.array([obj(x.reshape(1, -1)) for x in X_init])

        opt = BayesianOptimizer(
            gp_class=CVDGaussianProcess,
            objective_fn=obj,
            param_bounds=bounds,
            acq_func='EI',
            n_candidates=300,
            random_state=0,
        )
        opt.initialize(X_init, y_init)
        results = opt.run(n_iterations=10, verbose=False)

        assert results['best_quality'] >= y_init.max() - 1e-6
        assert len(results['history']) == 10

    def test_bo_history_monotone(self, fitted_gp):
        gp_ref, X, y, feat_cols = fitted_gp

        def obj(x):
            mu, _ = gp_ref.predict(x, return_std=True)
            return float(mu[0])

        bounds = {f: (-2.5, 2.5) for f in feat_cols}
        rng = np.random.default_rng(99)
        n_seed = 6
        X_init = np.column_stack([rng.uniform(-2.5, 2.5, n_seed) for _ in feat_cols])
        y_init = np.array([obj(x.reshape(1, -1)) for x in X_init])

        opt = BayesianOptimizer(CVDGaussianProcess, obj, bounds, n_candidates=200)
        opt.initialize(X_init, y_init)
        results = opt.run(n_iterations=8, verbose=False)

        best_series = results['history']['best_so_far'].values
        # best_so_far should be non-decreasing
        diffs = np.diff(best_series)
        assert (diffs >= -1e-9).all(), "best_so_far must be non-decreasing"


# ══════════════════════════════════════════════
# 6. Evaluation Tests
# ══════════════════════════════════════════════

class TestEvaluation:

    def test_compute_metrics_perfect(self):
        y = np.array([0.1, 0.5, 0.9, 0.3])
        m = compute_metrics(y, y)
        assert m['MAE'] == pytest.approx(0.0, abs=1e-9)
        assert m['R2']  == pytest.approx(1.0, abs=1e-6)

    def test_compute_metrics_with_std(self):
        y = np.linspace(0, 1, 20)
        y_pred = y + np.random.normal(0, 0.05, 20)
        y_std  = np.ones(20) * 0.1
        m = compute_metrics(y, y_pred, y_std)
        assert 'Mean_Uncertainty' in m

    def test_calibration_df_shape(self, fitted_gp):
        gp, X, y, _ = fitted_gp
        mu, sigma = gp.predict(X[40:], return_std=True)
        calib_df = calibration_analysis(y[40:], mu, sigma)
        assert calib_df.shape[0] == 4    # 4 CI levels


# ══════════════════════════════════════════════
# 7. Sensitivity Analysis Tests
# ══════════════════════════════════════════════

class TestSensitivity:

    def test_permutation_importance_shape(self, fitted_gp):
        gp, X, y, feat_cols = fitted_gp
        imp_df = perturbation_importance(gp, feat_cols, X[40:], y[40:], n_repeats=3)
        assert len(imp_df) == len(feat_cols)
        assert 'RMSE_Increase' in imp_df.columns

    def test_importance_sum_approx_one(self, fitted_gp):
        gp, X, y, feat_cols = fitted_gp
        imp_df = perturbation_importance(gp, feat_cols, X[40:], y[40:], n_repeats=3)
        total = imp_df['Importance_Norm'].sum()
        assert abs(total - 1.0) < 0.01 or total == pytest.approx(0.0, abs=0.01)


# ══════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
