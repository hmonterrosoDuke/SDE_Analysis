import numpy as np

class Model:
    """Base class for drift models."""
    
    def __init__(self, upper_trim = None, use_weights = False, fit_log = False):
        self.params = {}
        self.fitted = False
        self.fit_log = fit_log
        self.use_weights = use_weights
        self.upper_trim = upper_trim

    def _preprocess(self, edges, F):

        mask = np.isfinite(edges) & np.isfinite(F)
        x = edges[mask]
        y = F[mask]
        
        # Trim upper tail
        if self.upper_trim is not None:
            y_max = np.quantile(x, self.upper_trim)
            trim_mask = x <= y_max
            x = x[trim_mask]
            y = y[trim_mask]
        
        # Compute weights
        if self.use_weights:
            # Weight inversely by magnitude — equal importance across scales
            weights = 1 / (np.abs(y) + 1e-10)
        else:
            weights = np.ones_like(y)
        
        return x, y, weights
    
    def fit(self, edges, F):
        """Fit the model to data. Override in subclass."""
        raise NotImplementedError
    
    def evaluate(self, y):
        """Evaluate the model at y values. Override in subclass."""
        raise NotImplementedError
    
    def __repr__(self):
        if self.fitted:
            param_str = ', '.join([f'{k}={v:.3e}' for k, v in self.params.items()])
            return f"{self.__class__.__name__}({param_str})"
        return f"{self.__class__.__name__}(not fitted)"
    
def get_equilibrium(edges, F1):
    """Get equilibrium where drift crosses zero."""
    mask = np.isfinite(edges) & np.isfinite(F1)
    x = edges[mask]
    y = F1[mask]
    
    # Find where F1 changes sign (positive to negative)
    sign_change = np.where((y[:-1] > 0) & (y[1:] < 0))[0]
    
    if len(sign_change) > 0:
        # Linear interpolation to find exact crossing
        i = sign_change[0]
        x1, x2 = x[i], x[i+1]
        y1, y2 = y[i], y[i+1]
        y_eq = x1 - y1 * (x2 - x1) / (y2 - y1)
        return y_eq
    else:
        # Fallback: where |F1| is minimum
        return x[np.argmin(np.abs(y))]
    
    
def get_equilibrium_log(edges, F1, lower_trim=None):
    """Get equilibrium where drift crosses zero, anchored to the right of the peak."""
    mask = np.isfinite(edges) & np.isfinite(F1)
    x = edges[mask]
    y = F1[mask]

    if lower_trim is not None:
        x_min = np.quantile(x, lower_trim)
        trim_mask = x >= x_min
        x = x[trim_mask]
        y = y[trim_mask]

    # Find peak first
    peak_idx = np.argmax(y)

    # Only look for zero crossing to the right of peak
    y_right = y[peak_idx:]
    x_right = x[peak_idx:]

    sign_change = np.where((y_right[:-1] > 0) & (y_right[1:] < 0))[0]

    if len(sign_change) > 0:
        i = sign_change[0]
        x1, x2 = x_right[i], x_right[i+1]
        y1, y2 = y_right[i], y_right[i+1]
        y_eq = x1 - y1 * (x2 - x1) / (y2 - y1)
        return y_eq
    else:
        return x[np.argmin(np.abs(y))]
