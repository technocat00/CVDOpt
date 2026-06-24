"""
baseline_comparison.py
======================
Compare Bayesian Optimization against Random Search and Grid Search.
Demonstrates BO's sample efficiency advantage.
"""

import numpy as np
import pandas as pd
from itertools import product as cartesian_product
import time
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Random Search
# ─────────────────────────────────────────────

def random_search(objective_fn, param_bounds: dict, n_trials: int = 50,
                  random_state: int = 42) -> dict:
    """
    Randomly sample synthesis parameters and evaluate.
    No model, no intelligence — pure Monte Carlo.
    """
    rng = np.random.default_rng(random_state)
    n_params = len(param_bounds)
    bounds = list(param_bounds.values())

    history = []
    best_y = -np.inf
    best_x = None
    t0 = time.time()

    for trial in range(1, n_trials + 1):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        y = float(objective_fn(x.reshape(1, -1)))

        if y > best_y:
            best_y = y
            best_x = x.copy()

        history.append({
            'trial':        trial,
            'y':            round(y, 5),
            'best_so_far':  round(best_y, 5),
        })

    elapsed = time.time() - t0
    return {
        'method':       'Random Search',
        'best_quality': best_y,
        'best_x':       best_x,
        'n_trials':     n_trials,
        'time_s':       round(elapsed, 2),
        'history':      pd.DataFrame(history),
    }


# ─────────────────────────────────────────────
# 2. Grid Search
# ─────────────────────────────────────────────

def grid_search(objective_fn, param_bounds: dict, n_points_per_dim: int = 5,
                max_evals: int = 200) -> dict:
    """
    Uniform grid over parameter space.
    Suffers from the curse of dimensionality — scales as n^d.
    """
    bounds = list(param_bounds.values())
    n_dims = len(bounds)

    grids = [np.linspace(lo, hi, n_points_per_dim) for lo, hi in bounds]
    all_points = list(cartesian_product(*grids))

    # Truncate if too large
    if len(all_points) > max_evals:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_points), max_evals, replace=False)
        all_points = [all_points[i] for i in indices]
        print(f"[Grid] Subsampled to {max_evals} of {n_points_per_dim**n_dims} grid points.")

    history = []
    best_y = -np.inf
    best_x = None
    t0 = time.time()

    for trial, x in enumerate(all_points, 1):
        x_arr = np.array(x)
        y = float(objective_fn(x_arr.reshape(1, -1)))

        if y > best_y:
            best_y = y
            best_x = x_arr.copy()

        history.append({
            'trial':        trial,
            'y':            round(y, 5),
            'best_so_far':  round(best_y, 5),
        })

    elapsed = time.time() - t0
    return {
        'method':       'Grid Search',
        'best_quality': best_y,
        'best_x':       best_x,
        'n_trials':     len(all_points),
        'time_s':       round(elapsed, 2),
        'history':      pd.DataFrame(history),
    }


# ─────────────────────────────────────────────
# 3. Literature / Manual Baseline
# ─────────────────────────────────────────────

def literature_baseline(objective_fn, typical_conditions: np.ndarray) -> dict:
    """
    Evaluate a set of commonly reported CVD conditions from the literature.
    Represents what a researcher would try without optimization.
    """
    history = []
    best_y = -np.inf
    best_x = None

    for i, x in enumerate(typical_conditions, 1):
        y = float(objective_fn(x.reshape(1, -1)))
        if y > best_y:
            best_y = y
            best_x = x.copy()
        history.append({'trial': i, 'y': round(y, 5), 'best_so_far': round(best_y, 5)})

    return {
        'method':       'Literature Baseline',
        'best_quality': best_y,
        'best_x':       best_x,
        'n_trials':     len(typical_conditions),
        'history':      pd.DataFrame(history),
    }


# ─────────────────────────────────────────────
# 4. Comparison Runner
# ─────────────────────────────────────────────

def run_comparison(bo_results: dict, objective_fn, param_bounds: dict,
                   n_trials: int = 50, random_state: int = 42,
                   typical_conditions: np.ndarray = None) -> pd.DataFrame:
    """
    Run all baselines and compile a comparison table.
    """
    print("\n" + "="*55)
    print("  Baseline Comparison")
    print("="*55)

    results_list = []

    # ── BO (already computed) ──
    bo_hist = bo_results['history']
    results_list.append({
        'Method':       'Bayesian Optimization (EI)',
        'Best Quality': round(bo_results['best_quality'], 4),
        'Trials Used':  int(bo_hist.shape[0]),
        'Trials to 90% of Best': _trials_to_fraction(bo_hist, 0.90),
        'Trials to 95% of Best': _trials_to_fraction(bo_hist, 0.95),
    })

    # ── Random Search ──
    print("[Baseline] Running Random Search...")
    rs = random_search(objective_fn, param_bounds, n_trials=n_trials,
                       random_state=random_state)
    results_list.append({
        'Method':       'Random Search',
        'Best Quality': round(rs['best_quality'], 4),
        'Trials Used':  rs['n_trials'],
        'Trials to 90% of Best': _trials_to_fraction(rs['history'], 0.90),
        'Trials to 95% of Best': _trials_to_fraction(rs['history'], 0.95),
    })

    # ── Grid Search ──
    print("[Baseline] Running Grid Search...")
    n_dims = len(param_bounds)
    pts_per_dim = max(2, int(n_trials ** (1.0 / n_dims)))
    gs = grid_search(objective_fn, param_bounds, n_points_per_dim=pts_per_dim,
                     max_evals=n_trials)
    results_list.append({
        'Method':       'Grid Search',
        'Best Quality': round(gs['best_quality'], 4),
        'Trials Used':  gs['n_trials'],
        'Trials to 90% of Best': _trials_to_fraction(gs['history'], 0.90),
        'Trials to 95% of Best': _trials_to_fraction(gs['history'], 0.95),
    })

    # ── Literature baseline ──
    if typical_conditions is not None:
        print("[Baseline] Evaluating Literature Conditions...")
        lit = literature_baseline(objective_fn, typical_conditions)
        results_list.append({
            'Method':       'Literature Baseline',
            'Best Quality': round(lit['best_quality'], 4),
            'Trials Used':  lit['n_trials'],
            'Trials to 90% of Best': _trials_to_fraction(lit['history'], 0.90),
            'Trials to 95% of Best': _trials_to_fraction(lit['history'], 0.95),
        })

    summary_df = pd.DataFrame(results_list)
    print("\n" + summary_df.to_string(index=False))

    # Collect all histories for plotting
    histories = {
        'Bayesian Optimization': bo_results['history'].rename(columns={'best_so_far': 'best'}),
        'Random Search':         rs['history'].rename(columns={'best_so_far': 'best'}),
        'Grid Search':           gs['history'].rename(columns={'best_so_far': 'best'}),
    }
    if typical_conditions is not None:
        histories['Literature'] = lit['history'].rename(columns={'best_so_far': 'best'})

    return summary_df, histories, rs, gs


def _trials_to_fraction(history: pd.DataFrame, fraction: float) -> int:
    """Number of trials to reach `fraction` of the final best quality."""
    # Support both 'best_so_far' (BO) and 'best' (baselines) column names
    best_col  = 'best_so_far' if 'best_so_far' in history.columns else 'best'
    trial_col = 'iteration'   if 'iteration'   in history.columns else 'trial'

    final_best = history[best_col].iloc[-1]
    target = fraction * final_best
    mask = history[best_col] >= target
    if mask.any():
        return int(history.loc[mask, trial_col].iloc[0])
    return int(history[trial_col].iloc[-1])


# ─────────────────────────────────────────────
# CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Simple 2D Rosenbrock-like objective for quick demo
    def toy_objective(x):
        return float(-((1 - x[0, 0])**2 + 100*(x[0, 1] - x[0, 0]**2)**2) / 1000 + 1)

    bounds = {'x0': (-2, 2), 'x1': (-1, 3)}
    rs = random_search(toy_objective, bounds, n_trials=40)
    gs = grid_search(toy_objective, bounds, n_points_per_dim=8, max_evals=40)

    print(f"\nRandom Search best: {rs['best_quality']:.4f}")
    print(f"Grid Search best:   {gs['best_quality']:.4f}")
