"""
main.py
=======
End-to-end CVD synthesis optimization pipeline.
Run:  python main.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import (
    generate_raw_cvd_data, preprocess_pipeline, TARGET_COLS
)
from feature_engineering import (
    add_engineered_features, build_quality_score, get_feature_columns
)
from gpr_model import CVDGaussianProcess, calibration_check
from bayesian_optimization import BayesianOptimizer
from baseline_comparison import run_comparison
from evaluation import (
    train_test_evaluation, kfold_evaluation,
    calibration_analysis, learning_curve, compute_metrics
)
from sensitivity_analysis import (
    sensitivity_report, partial_dependence_all, interaction_map_2d
)
from visualization import (
    plot_bo_convergence, plot_baseline_comparison,
    plot_predicted_vs_actual, plot_temp_pressure_heatmap,
    plot_uncertainty_map, plot_feature_importance,
    plot_calibration, plot_partial_dependence_grid,
    plot_quality_distribution, plot_correlation_heatmap,
    plot_dashboard
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════
def banner(msg: str):
    print(f"\n{'═'*58}\n  {msg}\n{'═'*58}")


# ══════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    # ─────────────────────────────────────────
    # STEP 1  Data generation & preprocessing
    # ─────────────────────────────────────────
    banner("STEP 1 — Data Generation & Preprocessing")
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    raw_df = generate_raw_cvd_data(n_samples=150, random_state=42)
    raw_df.to_csv('data/raw_cvd_data.csv', index=False)

    proc_df, scaler, t_scalers = preprocess_pipeline(raw_df)

    # ─────────────────────────────────────────
    # STEP 2  Feature engineering
    # ─────────────────────────────────────────
    banner("STEP 2 — Feature Engineering")
    proc_df = add_engineered_features(proc_df)
    proc_df = build_quality_score(proc_df)
    proc_df.to_csv('data/processed_cvd_data.csv', index=False)

    feat_cols = get_feature_columns(proc_df)
    X = proc_df[feat_cols].values
    y = proc_df['quality_score'].values

    print(f"\nFeatures ({len(feat_cols)}): {feat_cols}")
    print(f"Quality score: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")

    # Visualise quality distribution and correlations
    plot_quality_distribution(y)
    plot_cols = feat_cols[:8] + ['quality_score']
    available = [c for c in plot_cols if c in proc_df.columns]
    plot_correlation_heatmap(proc_df, available)

    # ─────────────────────────────────────────
    # STEP 3  GPR model evaluation
    # ─────────────────────────────────────────
    banner("STEP 3 — Gaussian Process Regression Evaluation")
    gp_eval = CVDGaussianProcess(kernel_type='matern', n_restarts=5)
    metrics, y_test, y_pred, y_std, X_train, X_test, y_train = \
        train_test_evaluation(gp_eval, X, y, test_size=0.2)

    r2 = r2_score(y_test, y_pred)
    print(f"\nTrain/Test metrics: {metrics}")

    # K-fold CV
    cv_df, cv_agg = kfold_evaluation(CVDGaussianProcess, X, y, k=5)

    # Learning curve
    lc_df = learning_curve(CVDGaussianProcess, X, y)

    # Calibration
    calib_df = calibration_analysis(y_test, y_pred, y_std)

    plot_predicted_vs_actual(y_test, y_pred, y_std, r2=r2)
    plot_calibration(calib_df)

    # ─────────────────────────────────────────
    # STEP 4  Sensitivity analysis
    # ─────────────────────────────────────────
    banner("STEP 4 — Sensitivity / Feature Importance")
    gp_full = CVDGaussianProcess(kernel_type='matern', n_restarts=5).build()
    gp_full.fit(X_train, y_train, feature_names=feat_cols)

    imp_df = sensitivity_report(gp_full, feat_cols, X_test, y_test)
    plot_feature_importance(imp_df)

    # Partial dependence (first 6 features)
    pdp_feats = feat_cols[:6]
    pdp_idx   = [feat_cols.index(f) for f in pdp_feats]
    pdp_dict  = partial_dependence_all(gp_full, X_test, pdp_feats)
    plot_partial_dependence_grid(pdp_dict, ncols=3)

    # 2D interaction: temperature × log_pressure
    temp_idx  = feat_cols.index('temperature')  if 'temperature'  in feat_cols else 0
    press_idx = feat_cols.index('log_pressure') if 'log_pressure' in feat_cols else 1
    plot_temp_pressure_heatmap(gp_full, X_test, temp_idx=temp_idx, press_idx=press_idx)
    plot_uncertainty_map(gp_full, X_test, feat_i=temp_idx, feat_j=press_idx)

    # ─────────────────────────────────────────
    # STEP 5  Bayesian Optimization
    # ─────────────────────────────────────────
    banner("STEP 5 — Bayesian Optimization")

    # Build a surrogate objective from the reference GP
    ref_gp = CVDGaussianProcess(kernel_type='matern', n_restarts=5).build()
    ref_gp.fit(X, y, feature_names=feat_cols)

    def simulated_experiment(x: np.ndarray) -> float:
        """Simulate a CVD experiment (GP surrogate + noise)."""
        mu, sigma = ref_gp.predict(x, return_std=True)
        noise = np.random.normal(0, 0.02)
        return float(np.clip(mu[0] + noise, 0, 1))

    # Parameter bounds (standardised feature space)
    param_bounds = {f: (-2.5, 2.5) for f in feat_cols}

    # Seed with 12 random observations
    rng = np.random.default_rng(0)
    n_seed = 12
    X_seed = np.column_stack([rng.uniform(-2.5, 2.5, n_seed)
                               for _ in range(len(feat_cols))])
    y_seed = np.array([simulated_experiment(x.reshape(1, -1)) for x in X_seed])

    optimizer = BayesianOptimizer(
        gp_class=CVDGaussianProcess,
        objective_fn=simulated_experiment,
        param_bounds=param_bounds,
        acq_func='EI',
        xi=0.01,
        n_candidates=2000,
        random_state=42,
    )
    optimizer.initialize(X_seed, y_seed)
    bo_results = optimizer.run(n_iterations=35, verbose=True)

    plot_bo_convergence(bo_results['history'], n_init=n_seed)

    print(f"\n[BO] Best quality found: {bo_results['best_quality']:.4f}")
    print(f"[BO] Found at iteration: {bo_results['best_iteration']}")

    # ─────────────────────────────────────────
    # STEP 6  Baseline comparison
    # ─────────────────────────────────────────
    banner("STEP 6 — Baseline Comparison")
    n_trials = int(bo_results['history'].shape[0])
    summary_df, histories, rs_res, gs_res = run_comparison(
        bo_results, simulated_experiment, param_bounds,
        n_trials=n_trials, random_state=42
    )
    plot_baseline_comparison(histories)
    summary_df.to_csv('results/baseline_comparison.csv', index=False)
    print(f"\nComparison saved → results/baseline_comparison.csv")

    # ─────────────────────────────────────────
    # STEP 7  Summary dashboard
    # ─────────────────────────────────────────
    banner("STEP 7 — Summary Dashboard")
    plot_dashboard(
        bo_history=bo_results['history'],
        y_true=y_test,
        y_pred=y_pred,
        y_std=y_std,
        importance_df=imp_df,
        histories=histories,
        r2=r2
    )

    # ─────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────
    banner("PIPELINE COMPLETE")
    elapsed = time.time() - t_start
    print(f"  Total time         : {elapsed:.1f}s")
    print(f"  Dataset size       : {len(proc_df)} samples × {len(feat_cols)} features")
    print(f"  GPR  R²            : {r2:.4f}")
    print(f"  GPR  RMSE          : {metrics['RMSE']}")
    print(f"  BO best quality    : {bo_results['best_quality']:.4f}")
    print(f"  Random best quality: {rs_res['best_quality']:.4f}")
    print(f"  Grid best quality  : {gs_res['best_quality']:.4f}")
    print(f"\n  Figures → results/figures/")
    print(f"  Data    → data/")
    print(f"{'═'*58}")


if __name__ == '__main__':
    main()
