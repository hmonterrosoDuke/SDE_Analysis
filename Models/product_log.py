import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from .model import Model, get_equilibrium_log

import numpy as np
from scipy.optimize import curve_fit
from .model import Model, get_equilibrium


class AsymmetricDriftModel(Model):
    """
    Asymmetric drift model for log-space stage data:
    
        F1(z) = A * (z - z_eq) * exp(-beta_l * (z - z_eq)^2)   for z < z_eq
        F1(z) = m * (z - z_eq)                                   for z >= z_eq
    
    Left side: Gaussian derivative shape — sharp positive peak
    Right side: linear decay — persistent negative restoring force
    
    Parameters
    ----------
    A      : amplitude of left-side hump
    beta_l : width of left-side Gaussian
    m      : linear slope on right side (should be negative)
    """

    def __init__(self, upper_trim=None, lower_trim=None, use_weights=False):
        super().__init__(upper_trim=upper_trim, use_weights=use_weights)
        self.lower_trim = lower_trim

    def fit(self, edges, F1):
        y_eq = get_equilibrium(edges, F1)
        x, y, weights = self._preprocess(edges, F1)

        # Apply lower trim
        if self.lower_trim is not None:
            y_min = np.quantile(x, self.lower_trim)
            mask = x >= y_min
            x = x[mask]
            y = y[mask]
            weights = weights[mask]

        def model(z, A, beta_l, m):
            diff = z - y_eq
            left  = A * diff * np.exp(-beta_l * diff**2)
            right = m * diff
            return np.where(diff < 0, left, right)

        sigma = 1 / weights

        p0_candidates = [
            [1e-6,  1.0, -1e-6],
            [1e-5,  0.5, -1e-5],
            [1e-6,  2.0, -1e-7],
            [1e-5,  1.5, -1e-6],
            [1e-6,  0.5, -1e-7],
        ]

        best_popt, best_cost = None, np.inf
        for p0 in p0_candidates:
            try:
                popt, _ = curve_fit(
                    model, x, y,
                    p0=p0,
                    sigma=sigma,
                    bounds=([0, 0.01, -1], [1, 20, 0]),
                    maxfev=10000
                )
                cost = np.sum(((model(x, *popt) - y) * weights)**2)
                if cost < best_cost:
                    best_popt, best_cost = popt, cost
            except Exception:
                continue

        if best_popt is None:
            raise RuntimeError("All initializations failed")

        A, beta_l, m = best_popt

        self.fitted = True
        self.params = {
            'y_eq': y_eq,
            'A': A,
            'beta_l': beta_l,
            'm': m,
            'z_peak': y_eq - 1 / np.sqrt(2 * beta_l),  # location of positive peak
        }

    def evaluate(self, z):
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        z = np.asarray(z, dtype=float)
        y_eq  = self.params['y_eq']
        A     = self.params['A']
        beta_l = self.params['beta_l']
        m     = self.params['m']

        diff  = z - y_eq
        left  = A * diff * np.exp(-beta_l * diff**2)
        right = m * diff

        return np.where(diff < 0, left, right)