import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from .model import Model, get_equilibrium

class ProductModel(Model):
    """
    Smooth product model:
    F1 = m*(y - y_eq) * (1 + a*|y - y_eq|^(alpha-1))
    
    Linear near y_eq, power law far from y_eq.
    """
    def __init__(self, upper_trim=None, use_weights=False):
        super().__init__(upper_trim=upper_trim, use_weights = use_weights)

    def fit(self, edges, F1):
        y_eq = get_equilibrium(edges, F1)
        x, y, weights = self._preprocess(edges, F1)

        def model(y_arr, m, a, alpha):
            diff = y_arr - y_eq
            return m * diff * (1 + a * np.abs(diff)**(alpha - 1))

        sigma = 1 / weights

        # Try multiple initializations, keep best
        p0_candidates = [
            [-1e-4, 0.01, 3.0],
            [-1e-3, 0.1,  4.0],
            [-1e-2, 1.0,  5.0],
            [-1e-4, 0.01, 7.0],
            [-1e-3, 0.5,  2.0],
        ]

        best_popt, best_cost = None, np.inf
        for p0 in p0_candidates:
            try:
                popt, _ = curve_fit(
                    model, x, y,
                    p0=p0,
                    sigma=sigma,
                    bounds=([-5, -2, 1.1], [3, 15, 15]),
                    maxfev=5000
                )
                cost = np.sum(((model(x, *popt) - y) * weights) ** 2)
                if cost < best_cost:
                    best_popt, best_cost = popt, cost
            except Exception:
                continue

        if best_popt is None:
            raise RuntimeError("All initializations failed")

        m, a, alpha = best_popt
        self.params = {
            'y_eq': y_eq,
            'm': m,
            'a': a,
            'alpha': alpha
        }
        self.fitted = True
        return self
    
    def evaluate(self, y):
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y_eq = self.params['y_eq']
        m = self.params['m']
        a = self.params['a']
        alpha = self.params['alpha']
        
        y = np.asarray(y)
        diff = y - y_eq
        
        return m * diff * (1 + a * np.abs(diff)**(alpha - 1))