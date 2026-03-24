import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self, site_id, results, symlog=True, linthresh=1e-6):
        self.site_id = site_id
        self.symlog = symlog
        self.linthresh = linthresh

    def plot(self, results, fit, figsize=(10, 6), moment = 'F1'):
        y_fit, model_fit = fit
        edges = results[self.site_id]['KM_results']['edges']
        moments = results[self.site_id]['KM_results'][moment]

        fig, ax = plt.subplots(figsize=figsize)
        
        # Data
        ax.plot(edges, moments, 'k.', alpha=0.7, label='Data')
        ax.plot(y_fit, model_fit, 'r-', lw=3, alpha=0.7, label='Fit')
        
        
        # Formatting
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        if self.symlog:
            ax.set_yscale('symlog', linthresh=self.linthresh)
        ax.set_xlabel('y (normalized)')
        ax.set_ylabel('$F_1$ (Drift)')
        ax.set_title(self.site_id)
        ax.legend()
        
        return fig, ax