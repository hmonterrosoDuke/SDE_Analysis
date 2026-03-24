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
            diff = y_arr - y_eq  # y_eq captured from outer scope
            return m * diff * (1 + a * np.abs(diff)**(alpha - 1))
        
        sigma = 1/weights
    
        popt, _ = curve_fit(model, x, y, p0=[-1e-4, 0.01, 2.0], sigma = sigma,
                            bounds=([-5, 0, 0.5], [2, 10, 8]), maxfev=5000)
        m, a, alpha = popt

        
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