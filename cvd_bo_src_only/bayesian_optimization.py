"""
bayesian_optimization.py
========================
Bayesian Optimization loop for CVD synthesis parameter selection.
Acquisition functions: Expected Improvement (EI), PI, UCB.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from typing import Callable, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Acquisition Functions
# ─────────────────────────────────────────────

def expected_improvement(X_candidates: np.ndarray, gp_model, y_best: float,
                          xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement (EI) — recommended for CVD optimization.

    EI(x) = E[max(f(x) − y_best, 0)]
           = (μ − y_best − ξ) * Φ(Z) + σ * φ(Z)
    where Z = (μ − y_best − ξ) / σ

    xi controls exploration-exploitation balance (larger → more exploration).
    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    sigma = sigma.reshape(-1)
    mu = mu.reshape(-1)

    imp = mu - y_best - xi
    Z = np.where(sigma > 1e-9, imp / sigma, 0.0)

    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def probability_of_improvement(X_candidates: np.ndarray, gp_model, y_best: float,
                                 xi: float = 0.01) -> np.ndarray:
    """
    Probability of Improvement (PI).
    PI(x) = P(f(x) > y_best + ξ) = Φ((μ − y_best − ξ) / σ)
    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    Z = np.where(sigma > 1e-9, (mu - y_best - xi) / sigma, 0.0)
    return norm.cdf(Z)


def upper_confidence_bound(X_candidates: np.ndarray, gp_model, y_best: float,
                             kappa: float = 2.576) -> np.ndarray:
    """
    Upper Confidence Bound (UCB).
    UCB(x) = μ(x) + κ * σ(x)

    kappa: exploration weight (higher → more exploration).
    """
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    return mu + kappa * sigma


ACQUISITION_FUNCTIONS = {
    'EI':  expected_improvement,
    'PI':  probability_of_improvement,
    'UCB': upper_confidence_bound,
}


# ─────────────────────────────────────────────
# 2. Candidate Space Generator
# ─────────────────────────────────────────────

def build_candidate_grid(param_bounds: dict, n_grid: int = 2000,
                          random_state: int = 42) -> np.ndarray:
    """
    Generate a random candidate set (Latin hypercube-like) within parameter bounds.

    param_bounds: dict {name: (low, high)} in the SCALED feature space.
    Returns array of shape (n_grid, n_params).
    """
    rng = np.random.default_rng(random_state)
    n_params = len(param_bounds)
    bounds = list(param_bounds.values())

    candidates = np.column_stack([
        rng.uniform(lo, hi, n_grid) for lo, hi in bounds
    ])
    return candidates


# ─────────────────────────────────────────────
# 3. Single BO Step
# ─────────────────────────────────────────────

def suggest_next_experiment(gp_model, X_observed: np.ndarray, y_observed: np.ndarray,
                             param_bounds: dict, acq_func: str = 'EI',
                             n_candidates: int = 5000, xi: float = 0.01,
                             kappa: float = 2.576, random_state: int = None) -> tuple:
    """
    Given the current GP surrogate and observations, suggest the next
    synthesis parameter set to evaluate.

    Returns: (best_candidate_array, acquisition_value, all_acq_values)
    """
    rs = random_state or np.random.randint(0, 9999)
    X_candidates = build_candidate_grid(param_bounds, n_grid=n_candidates, random_state=rs)

    y_best = float(np.max(y_observed))
    acq_fn = ACQUISITION_FUNCTIONS.get(acq_func.upper())
    if acq_fn is None:
        raise ValueError(f"Unknown acquisition function: {acq_func}. "
                         f"Choose from {list(ACQUISITION_FUNCTIONS)}")

    acq_values = acq_fn(X_candidates, gp_model, y_best, xi=xi)

    best_idx = np.argmax(acq_values)
    return X_candidates[best_idx], acq_values[best_idx], acq_values


# ─────────────────────────────────────────────
# 4. Full BO Loop
# ─────────────────────────────────────────────

class BayesianOptimizer:
    """
    Bayesian Optimization loop for CVD synthesis.

    Usage:
        optimizer = BayesianOptimizer(gp_class, objective_fn, param_bounds)
        optimizer.initialize(X_init, y_init)
        results = optimizer.run(n_iterations=30)
    """

    def __init__(self, gp_class, objective_fn: Callable, param_bounds: dict,
                 acq_func: str = 'EI', xi: float = 0.01, kappa: float = 2.576,
                 n_candidates: int = 3000, random_state: int = 42):
        self.gp_class = gp_class
        self.objective_fn = objective_fn    # black-box function: x → quality_score
        self.param_bounds = param_bounds
        self.acq_func = acq_func
        self.xi = xi
        self.kappa = kappa
        self.n_candidates = n_candidates
        self.random_state = random_state

        self.X_observed = None
        self.y_observed = None
        self.gp = None

        self.history = []    # list of dicts per iteration

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray):
        """Seed the optimizer with initial observations."""
        self.X_observed = X_init.copy()
        self.y_observed = y_init.copy()
        print(f"[BO] Initialized with {len(y_init)} seed observations. "
              f"Best quality so far: {np.max(y_init):.4f}")
        return self

    def _fit_gp(self):
        self.gp = self.gp_class().build()
        self.gp.fit(self.X_observed, self.y_observed)

    def run(self, n_iterations: int = 30, verbose: bool = True):
        """
        Main optimization loop:
            Fit GP → Suggest next x via acquisition → Evaluate → Update.
        """
        if self.X_observed is None:
            raise RuntimeError("Call .initialize() first.")

        print(f"\n{'='*55}")
        print(f"  Bayesian Optimization  |  {self.acq_func}  |  {n_iterations} iters")
        print(f"{'='*55}")

        for iteration in range(1, n_iterations + 1):
            # 1. Fit GP surrogate on current observations
            self._fit_gp()

            # 2. Suggest next experiment
            x_next, acq_val, _ = suggest_next_experiment(
                self.gp, self.X_observed, self.y_observed,
                self.param_bounds, self.acq_func,
                n_candidates=self.n_candidates,
                xi=self.xi, kappa=self.kappa,
                random_state=self.random_state + iteration
            )

            # 3. Evaluate objective (simulate experiment)
            y_next = float(self.objective_fn(x_next.reshape(1, -1)))

            # 4. Update observations
            self.X_observed = np.vstack([self.X_observed, x_next])
            self.y_observed = np.append(self.y_observed, y_next)

            best_so_far = float(np.max(self.y_observed))
            self.history.append({
                'iteration':       iteration,
                'y_next':          round(y_next, 5),
                'best_so_far':     round(best_so_far, 5),
                'acq_value':       round(float(acq_val), 6),
                'n_observations':  len(self.y_observed),
            })

            if verbose and (iteration % 5 == 0 or iteration == 1):
                print(f"  Iter {iteration:3d} | quality = {y_next:.4f} | "
                      f"best = {best_so_far:.4f} | acq = {acq_val:.5f}")

        print(f"\n[BO] Done. Best quality: {np.max(self.y_observed):.4f} "
              f"at iteration {int(np.argmax(self.y_observed)) + 1}.")
        return self.get_results()

    def get_results(self) -> dict:
        best_idx = int(np.argmax(self.y_observed))
        return {
            'best_quality':    float(np.max(self.y_observed)),
            'best_x':          self.X_observed[best_idx],
            'best_iteration':  best_idx + 1,
            'history':         pd.DataFrame(self.history),
            'X_observed':      self.X_observed,
            'y_observed':      self.y_observed,
        }


# ─────────────────────────────────────────────
# 5. CLI / demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from gpr_model import CVDGaussianProcess
    from data_preprocessing import generate_raw_cvd_data, preprocess_pipeline
    from feature_engineering import (add_engineered_features, build_quality_score,
                                      get_feature_columns)

    # ── Build a surrogate objective from real data ──
    raw_df = generate_raw_cvd_data(120)
    proc_df, scaler, _ = preprocess_pipeline(raw_df)
    proc_df = add_engineered_features(proc_df)
    proc_df = build_quality_score(proc_df)
    feat_cols = get_feature_columns(proc_df)

    X_all = proc_df[feat_cols].values
    y_all = proc_df['quality_score'].values

    # Train a reference GP on all data to act as the black-box "experiment"
    ref_gp = CVDGaussianProcess().build()
    ref_gp.fit(X_all, y_all)

    def simulated_experiment(x):
        mu, _ = ref_gp.predict(x, return_std=True)
        return float(mu[0]) + np.random.normal(0, 0.02)  # add noise

    # ── Bounds (in standardised feature space ≈ [-3, 3]) ──
    n_features = X_all.shape[1]
    param_bounds = {f'f{i}': (-2.5, 2.5) for i in range(n_features)}

    # ── Seed observations (10 random) ──
    rng = np.random.default_rng(0)
    X_init = np.column_stack([rng.uniform(-2.5, 2.5, 10) for _ in range(n_features)])
    y_init = np.array([simulated_experiment(x.reshape(1, -1)) for x in X_init])

    # ── Run BO ──
    optimizer = BayesianOptimizer(
        gp_class=CVDGaussianProcess,
        objective_fn=simulated_experiment,
        param_bounds=param_bounds,
        acq_func='EI',
        n_candidates=2000,
    )
    optimizer.initialize(X_init, y_init)
    results = optimizer.run(n_iterations=20, verbose=True)

    print(f"\nFinal best quality: {results['best_quality']:.4f}")
    print(results['history'].tail(5).to_string(index=False))
