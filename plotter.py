import matplotlib.pyplot as plt
import numpy as np
import os
import math


class Plotter:

    def __init__(self, symlog=True, linthresh=1e-6, moment='F1'):
        self.symlog = symlog
        self.linthresh = linthresh
        self.moment = moment

    def _plot_to_ax(self, ax, site_id, results):
        site = results[site_id]
        edges  = site['KM_results']['edges']
        moments = site['KM_results'][self.moment]
        y_fit, model_fit = site['fit']

        ax.plot(edges, moments, 'k.', alpha=0.7, label='Data')
        ax.plot(y_fit, model_fit, 'r-', lw=3, alpha=0.7, label='Fit')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)

        if self.symlog:
            ax.set_yscale('symlog', linthresh=self.linthresh)

        ax.set_xlabel('y (normalized)')
        ax.set_ylabel(f'$F_1$ (Drift)')
        ax.set_title(site_id)
        ax.legend(fontsize=7)

    def plot(self, site_id, results, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_to_ax(ax, site_id, results)
        return fig, ax

    def plot_all(self, results, subplot_size=(4, 3)):
        sites_with_fit = [s for s in results if 'fit' in results[s]]

        if not sites_with_fit:
            print("No sites with a fit found.")
            return None

        n = len(sites_with_fit)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        figsize = (ncols * subplot_size[0], nrows * subplot_size[1])
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten()  # handle 1-row or 1-col edge cases

        for ax, site_id in zip(axes, sites_with_fit):
            self._plot_to_ax(ax, site_id, results)

        for ax in axes[n:]:  # hide leftover empty axes
            ax.set_visible(False)

        fig.tight_layout()
        return fig

    def save_all(self, results, folder, fmt='png', dpi=150):
        os.makedirs(folder, exist_ok=True)
        sites_with_fit = [s for s in results if 'fit' in results[s]]

        if not sites_with_fit:
            print("No sites with a fit found.")
            return

        for site_id in sites_with_fit:
            fig, _ = self.plot(site_id, results)
            path = os.path.join(folder, f"{site_id}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {path}")