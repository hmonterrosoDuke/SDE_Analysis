import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from .model import Model, get_equilibrium

class LinearPowerLaw(Model):
    """
    Piecewise drift model:
    F1 = m*(y - y_eq) for y < y_eq
    F1 = -a*(y - y_eq)^alpha for y >= y_eq
    """

    def get_linear_slope(self,x,y, y_eq):
        # Linear fit
        below = x < y_eq
        m = np.sum((x[below] - y_eq) * y[below]) / np.sum((x[below] - y_eq)**2)

        return m

    def fit_power_law(self,x,y,y_eq):
        # Above: power law with free alpha (allow alpha < 1)
        above = x >= y_eq
        x_above = x[above] - y_eq
        y_above = y[above]
        
        # logspace fit - powerlaw
        mask_fit = (x_above > 0.01) & (y_above < 0)
        log_x = np.log(x_above[mask_fit])
        log_y = np.log(-y_above[mask_fit])
        alpha, log_a = np.polyfit(log_x, log_y, 1)
        a = np.exp(log_a)

        return alpha, a

    def fit(self, edges, F1):
        """Fit F1 = m*(y - y_eq) for y < y_eq, F1 = -a*(y - y_eq)^alpha for y >= y_eq."""
        
        y_eq = get_equilibrium(edges, F1)
        
        mask = np.isfinite(edges) & np.isfinite(F1)
        x = edges[mask]
        y = F1[mask]
        
        m = self.get_linear_slope(x,y, y_eq)
        alpha, a = self.fit_power_law(x,y,y_eq)

        self.fitted = True
        
        self.params =  {
            'y_eq': y_eq,
            'm': m,
            'a': a,
            'alpha': alpha
        }

    def evaluate(self, y):
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y_eq = self.params['y_eq']
        m = self.params['m']
        a = self.params['a']
        alpha = self.params['alpha']
        
        y = np.asarray(y)
        result = np.zeros_like(y, dtype=float)
        
        below = y < y_eq
        above = y >= y_eq
        
        result[below] = m * (y[below] - y_eq)
        result[above] = -a * (y[above] - y_eq)**alpha
        
        return result
