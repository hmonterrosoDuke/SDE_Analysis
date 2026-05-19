import numpy as np
from scipy.optimize import curve_fit
from .model import Model


class GammaDiffusion(Model):
    """
    Diffusion model with unimodal (gamma-like) form:
        D(y) = C + a * y^beta * exp(-gamma * y)

    Peak location: y* = beta / gamma
    C allows for a non-zero floor and sub-floor decay at high y.
    """

    def fit(self, edges, D2):

        def _func(y, C, a, beta, gamma):
            return C + a * np.power(y, beta) * np.exp(-gamma * y)

        x, y, weights = self._preprocess(edges, D2)
        mask = (x > 0) & (y > 0)
        x_fit = x[mask]
        y_fit = y[mask]

        peak_idx = np.argmax(y_fit)
        y_peak = x_fit[peak_idx]
        C0 = y_fit[0]
        a0 = y_fit[peak_idx] - C0
        beta0 = 2.0
        gamma0 = beta0 / y_peak if y_peak > 0 else 0.1

        sigma = 1.0 / weights[mask] if self.use_weights else None

        popt, pcov = curve_fit(
            _func,
            x_fit,
            y_fit,
            p0=[C0, a0, beta0, gamma0],
            sigma=sigma,
            bounds=([-np.inf, 0, 0, 1e-6], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=10000,
        )

        C, a, beta, gamma = popt
        perr = np.sqrt(np.diag(pcov))

        self.fitted = True
        self.params = {
            'C': C,
            'a': a,
            'beta': beta,
            'gamma': gamma,
            'y_peak': beta / gamma,
        }
        self.pcov = pcov
        self.perr = {'C': perr[0], 'a': perr[1], 'beta': perr[2], 'gamma': perr[3]}

    def evaluate(self, y):
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        y = np.asarray(y, dtype=float)
        C = self.params['C']
        a = self.params['a']
        beta = self.params['beta']
        gamma = self.params['gamma']
        return C + a * np.power(y, beta) * np.exp(-gamma * y)