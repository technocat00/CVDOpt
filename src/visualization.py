"""
visualization.py
================
All plotting functions for the CVD BO project.
Saves figures to results/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = {
    'bo':     '#2563EB',
    'random': '#DC2626',
    'grid':   '#16A34A',
    'lit':    '#9333EA',
    'uncertainty': '#F59E0B',
    'predicted':   '#0EA5E9',
    'actual':      '#1E293B',
}
FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')


def _save(fig, name: str, dpi: int = 150):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 1. BO Convergence Curve
# ─────────────────────────────────────────────

def plot_bo_convergence(bo_history: pd.DataFrame, n_init: int = 10,
                        title: str = "Bayesian Optimization Convergence") -> str:
    """Shows best quality score improving over BO iterations."""
    fig, ax = plt.subplots(figsize=(9, 5))

    iters = bo_history['iteration'].values
    best  = bo_history['best_so_far'].values
    y_raw = bo_history['y_next'].values

    ax.scatter(iters, y_raw, alpha=0.5, s=30, color=COLORS['bo'],
               label='Observed quality', zorder=3)
    ax.plot(iters, best, color=COLORS['bo'], lw=2.5, label='Best so far', zorder=4)
    ax.axvline(x=n_init, color='grey', ls='--', lw=1.5, label=f'Seed ({n_init} obs)')

    ax.set(xlabel='Iteration', ylabel='Quality Score',
           title=title, xlim=(1, len(iters)))
    ax.legend()
    fig.tight_layout()
    return _save(fig, 'bo_convergence')


# ─────────────────────────────────────────────
# 2. Baseline Comparison Curves
# ─────────────────────────────────────────────

def plot_baseline_comparison(histories: dict) -> str:
    """Overlay BO vs Random vs Grid convergence curves."""
    color_map = {
        'Bayesian Optimization': COLORS['bo'],
        'Random Search':         COLORS['random'],
        'Grid Search':           COLORS['grid'],
        'Literature':            COLORS['lit'],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, hist in histories.items():
        col = color_map.get(name, 'black')
        trial_col = 'iteration' if 'iteration' in hist.columns else 'trial'
        best_col  = 'best_so_far' if 'best_so_far' in hist.columns else 'best'
        ax.plot(hist[trial_col].values, hist[best_col].values,
                label=name, color=col, lw=2.2)

    ax.set(xlabel='Number of Experiments', ylabel='Best Quality Score',
           title='Bayesian Optimization vs Baselines')
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return _save(fig, 'baseline_comparison')


# ─────────────────────────────────────────────
# 3. Predicted vs Actual
# ─────────────────────────────────────────────

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              y_std: np.ndarray = None,
                              r2: float = None) -> str:
    """Scatter of predicted vs actual quality, with ±1σ error bars."""
    fig, ax = plt.subplots(figsize=(6, 6))

    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05

    if y_std is not None:
        ax.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6,
                    color=COLORS['predicted'], ecolor='#94A3B8',
                    elinewidth=1, capsize=3, ms=5, label='Prediction ± 1σ')
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, color=COLORS['predicted'], s=40)

    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect fit')
    label = f"R² = {r2:.3f}" if r2 is not None else ""
    ax.text(0.05, 0.92, label, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    ax.set(xlabel='Actual Quality Score', ylabel='Predicted Quality Score',
           title='Predicted vs Actual', xlim=(lo, hi), ylim=(lo, hi))
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()
    return _save(fig, 'predicted_vs_actual')


# ─────────────────────────────────────────────
# 4. Temperature–Pressure Heatmap
# ─────────────────────────────────────────────

def plot_temp_pressure_heatmap(gp_model, X_background: np.ndarray,
                                temp_idx: int = 0, press_idx: int = 1,
                                n_grid: int = 40) -> str:
    """2D heatmap of predicted quality over temperature × log(pressure)."""
    temps   = np.linspace(-2.5, 2.5, n_grid)
    log_prs = np.linspace(-2.5, 2.5, n_grid)
    T_grid, P_grid = np.meshgrid(temps, log_prs)

    Z = np.zeros_like(T_grid)
    for i in range(n_grid):
        for j in range(n_grid):
            X_mod = X_background.mean(axis=0, keepdims=True).copy()
            X_mod = np.tile(X_mod, (1, 1))
            X_mod[0, temp_idx]  = T_grid[i, j]
            X_mod[0, press_idx] = P_grid[i, j]
            mu, _ = gp_model.predict(X_mod, return_std=True)
            Z[i, j] = float(mu[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(T_grid, P_grid, Z, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Predicted Quality Score')
    ax.set(xlabel='Temperature (scaled)', ylabel='Log Pressure (scaled)',
           title='Predicted Quality: Temperature × Pressure')
    fig.tight_layout()
    return _save(fig, 'temp_pressure_heatmap')


# ─────────────────────────────────────────────
# 5. Uncertainty Contour Map
# ─────────────────────────────────────────────

def plot_uncertainty_map(gp_model, X_background: np.ndarray,
                          feat_i: int = 0, feat_j: int = 1,
                          n_grid: int = 40) -> str:
    """Contour map of GP uncertainty (σ) — shows unexplored regions."""
    xi = np.linspace(-2.5, 2.5, n_grid)
    xj = np.linspace(-2.5, 2.5, n_grid)
    Xi, Xj = np.meshgrid(xi, xj)
    Z_std = np.zeros_like(Xi)

    for i in range(n_grid):
        for j in range(n_grid):
            X_mod = X_background.mean(axis=0, keepdims=True).copy()
            X_mod[0, feat_i] = Xi[i, j]
            X_mod[0, feat_j] = Xj[i, j]
            _, sigma = gp_model.predict(X_mod, return_std=True)
            Z_std[i, j] = float(sigma[0])

    # Overlay observed points
    obs_i = X_background[:, feat_i]
    obs_j = X_background[:, feat_j]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(Xi, Xj, Z_std, levels=20, cmap='YlOrRd')
    fig.colorbar(cf, ax=ax, label='Predictive Uncertainty (σ)')
    ax.scatter(obs_i, obs_j, c='blue', s=15, alpha=0.6, label='Observed points', zorder=5)
    ax.set(xlabel=f'Feature {feat_i} (scaled)', ylabel=f'Feature {feat_j} (scaled)',
           title='GP Uncertainty Map — Unexplored Regions (High σ)')
    ax.legend()
    fig.tight_layout()
    return _save(fig, 'uncertainty_contour_map')


# ─────────────────────────────────────────────
# 6. Feature Importance Bar Chart
# ─────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame,
                             value_col: str = 'Importance_Norm',
                             feature_col: str = 'Feature') -> str:
    """Horizontal bar chart of feature importances."""
    df = importance_df.sort_values(value_col, ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    bars = ax.barh(df[feature_col], df[value_col], color=COLORS['bo'], alpha=0.85)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.set(xlabel='Normalised Importance', title='Feature Importance\n(Permutation-based)')
    fig.tight_layout()
    return _save(fig, 'feature_importance')


# ─────────────────────────────────────────────
# 7. Calibration Plot
# ─────────────────────────────────────────────

def plot_calibration(calib_df: pd.DataFrame) -> str:
    """Expected vs observed coverage for uncertainty calibration."""
    fig, ax = plt.subplots(figsize=(6, 5))

    expected = calib_df['Expected Coverage'].values
    observed = calib_df['Observed Coverage'].values

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.scatter(expected, observed, s=80, color=COLORS['bo'], zorder=5)
    for i, row in calib_df.iterrows():
        ax.annotate(row['Confidence Level'],
                    (row['Expected Coverage'], row['Observed Coverage']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)

    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='green')
    ax.set(xlabel='Expected Coverage', ylabel='Observed Coverage',
           title='Uncertainty Calibration', xlim=(0.5, 1.05), ylim=(0.5, 1.05))
    ax.legend()
    fig.tight_layout()
    return _save(fig, 'calibration_plot')


# ─────────────────────────────────────────────
# 8. Partial Dependence Plots (grid)
# ─────────────────────────────────────────────

def plot_partial_dependence_grid(pdp_dict: dict, ncols: int = 3) -> str:
    """Grid of 1D partial dependence plots for each feature."""
    names = list(pdp_dict.keys())
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(names):
        df = pdp_dict[name]
        x  = df[name].values
        mu = df['mean_quality'].values
        sd = df['mean_std'].values
        axes[i].plot(x, mu, color=COLORS['bo'], lw=2)
        axes[i].fill_between(x, mu - sd, mu + sd, alpha=0.2, color=COLORS['bo'])
        axes[i].set(title=name, xlabel='Feature value (scaled)',
                    ylabel='Quality Score')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Partial Dependence Plots', fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, 'partial_dependence_grid')


# ─────────────────────────────────────────────
# 9. Quality Score Distribution
# ─────────────────────────────────────────────

def plot_quality_distribution(y: np.ndarray, title: str = "Quality Score Distribution") -> str:
    """Histogram + KDE of the composite quality scores in the dataset."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(y, bins=25, kde=True, ax=ax, color=COLORS['bo'], alpha=0.7)
    ax.axvline(np.mean(y), color='red', ls='--', lw=1.5, label=f'Mean = {np.mean(y):.3f}')
    ax.axvline(np.max(y),  color='green', ls='--', lw=1.5, label=f'Max = {np.max(y):.3f}')
    ax.set(xlabel='Quality Score', ylabel='Count', title=title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, 'quality_distribution')


# ─────────────────────────────────────────────
# 10. Correlation Heatmap
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, cols: list) -> str:
    """Heatmap of Pearson correlations among synthesis features and quality."""
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.8)))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.4, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=13)
    fig.tight_layout()
    return _save(fig, 'correlation_heatmap')


# ─────────────────────────────────────────────
# 11. Summary Dashboard
# ─────────────────────────────────────────────

def plot_dashboard(bo_history: pd.DataFrame, y_true: np.ndarray,
                   y_pred: np.ndarray, y_std: np.ndarray,
                   importance_df: pd.DataFrame, histories: dict,
                   r2: float = None) -> str:
    """4-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: BO convergence ──
    ax1 = fig.add_subplot(gs[0, 0])
    iters = bo_history['iteration'].values
    ax1.plot(iters, bo_history['best_so_far'].values, color=COLORS['bo'], lw=2.5)
    ax1.scatter(iters, bo_history['y_next'].values, alpha=0.4, s=20, color=COLORS['bo'])
    ax1.set(title='BO Convergence', xlabel='Iteration', ylabel='Quality Score')

    # ── Panel 2: Baseline comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    color_map2 = {
        'Bayesian Optimization': COLORS['bo'],
        'Random Search':         COLORS['random'],
        'Grid Search':           COLORS['grid'],
    }
    for name, hist in histories.items():
        col = color_map2.get(name, 'grey')
        trial_col = 'iteration' if 'iteration' in hist.columns else 'trial'
        best_col  = 'best_so_far' if 'best_so_far' in hist.columns else 'best'
        ax2.plot(hist[trial_col].values, hist[best_col].values, label=name, color=col, lw=2)
    ax2.set(title='BO vs Baselines', xlabel='Experiments', ylabel='Best Quality')
    ax2.legend(fontsize=9)

    # ── Panel 3: Predicted vs actual ──
    ax3 = fig.add_subplot(gs[1, 0])
    lo = min(y_true.min(), y_pred.min()) - 0.02
    hi = max(y_true.max(), y_pred.max()) + 0.02
    ax3.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.5,
                 color=COLORS['predicted'], ecolor='#94A3B8', ms=4)
    ax3.plot([lo, hi], [lo, hi], 'k--', lw=1.5)
    if r2 is not None:
        ax3.text(0.05, 0.9, f'R² = {r2:.3f}', transform=ax3.transAxes,
                 bbox=dict(fc='white', alpha=0.7))
    ax3.set(title='Predicted vs Actual', xlabel='Actual', ylabel='Predicted',
            xlim=(lo, hi), ylim=(lo, hi))

    # ── Panel 4: Feature importance ──
    ax4 = fig.add_subplot(gs[1, 1])
    df_imp = importance_df.sort_values('Importance_Norm', ascending=True).tail(10)
    ax4.barh(df_imp['Feature'], df_imp['Importance_Norm'], color=COLORS['bo'], alpha=0.8)
    ax4.set(title='Feature Importance', xlabel='Normalised Importance')

    fig.suptitle('CVD Synthesis Optimization — Summary Dashboard',
                 fontsize=15, fontweight='bold', y=1.01)
    return _save(fig, 'summary_dashboard')


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_preprocessing import generate_raw_cvd_data, preprocess_pipeline
    from feature_engineering import add_engineered_features, build_quality_score, get_feature_columns
    from gpr_model import CVDGaussianProcess
    from evaluation import train_test_evaluation, calibration_analysis
    from sensitivity_analysis import perturbation_importance, partial_dependence_all
    from sklearn.metrics import r2_score

    raw_df = generate_raw_cvd_data(120)
    proc_df, _, _ = preprocess_pipeline(raw_df)
    proc_df = add_engineered_features(proc_df)
    proc_df = build_quality_score(proc_df)
    feat_cols = get_feature_columns(proc_df)

    X = proc_df[feat_cols].values
    y = proc_df['quality_score'].values

    gp = CVDGaussianProcess(kernel_type='matern')
    _, y_test, y_pred, y_std, *extras = train_test_evaluation(gp, X, y)

    r2 = r2_score(y_test, y_pred)
    calib_df = calibration_analysis(y_test, y_pred, y_std)
    imp_df = perturbation_importance(gp, feat_cols)

    plot_predicted_vs_actual(y_test, y_pred, y_std, r2=r2)
    plot_feature_importance(imp_df)
    plot_calibration(calib_df)
    plot_quality_distribution(y)
    print("Demo plots saved to results/figures/")
