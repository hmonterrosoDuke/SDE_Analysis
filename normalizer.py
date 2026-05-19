"""
GPD-based normalizer for 15-minute river stage data.
 
Usage:
    norm = RiverStageNormalizer(return_period_years=5).fit(data)
    normalized = norm.transform(data)
    original = norm.inverse_transform(normalized)
"""
import numpy as np
from scipy import stats
 
 
SAMPLES_PER_DAY_15MIN = 96
SAMPLES_PER_YEAR_15MIN = 365.25 * SAMPLES_PER_DAY_15MIN  # ~35,064
 
 
def get_peaks(data, threshold, min_gap):
    """
    Decluster exceedances into independent peaks.
 
    Keeps the maximum of each run of above-threshold values, where runs
    are separated by at least `min_gap` consecutive below-threshold samples.
 
    """
    data = np.asarray(data)
    above = data > threshold
    peaks, cluster_vals, gap, in_cluster = [], [], 0, False
    for v, a in zip(data, above):
        if a:
            cluster_vals.append(v)
            gap = 0
            in_cluster = True
        elif in_cluster:
            gap += 1
            if gap >= min_gap:
                peaks.append(max(cluster_vals))
                cluster_vals, in_cluster = [], False
    if cluster_vals:
        peaks.append(max(cluster_vals))
    return np.array(peaks)
 
 
def fit_gpd(peaks, threshold):

    excesses = peaks - threshold
    shape, _, scale = stats.genpareto.fit(excesses, floc=0)
    return shape, scale
 
 
def return_level(threshold, shape, scale, zeta, N):

    if abs(shape) < 1e-6:
        return threshold + scale * np.log(N * zeta)
    return threshold + (scale / shape) * ((N * zeta) ** shape - 1)
 
 
class RiverStageNormalizer:
    """
    Fit a GPD tail model and normalize by a chosen return level.
 
    Parameters
    ----------
    return_period_years : float
        Target return period (default 5 years).
    threshold_quantile : float
        Quantile used to set the GPD threshold (default 0.95).
    min_gap_days : float
        Minimum days between independent flood peaks (default 5).
    samples_per_year : float
        Sampling frequency (default = 15-minute data).
    """
 
    def __init__(self, return_period_years=5, threshold_quantile=0.95,
                 min_gap_days=5, samples_per_year=SAMPLES_PER_YEAR_15MIN):
        self.return_period_years = return_period_years
        self.threshold_quantile = threshold_quantile
        self.min_gap_days = min_gap_days
        self.samples_per_year = samples_per_year
        # filled in by fit()
        self.threshold_ = None
        self.shape_ = None
        self.scale_ = None
        self.zeta_ = None
        self.return_level_ = None
        self.n_peaks_ = None
        self.peaks_per_year_ = None
 
    def fit(self, data):
        data = np.asarray(data)
        data = data[~np.isnan(data)]
 
        samples_per_day = self.samples_per_year / 365.25
        min_gap = int(round(self.min_gap_days * samples_per_day))
 
        self.threshold_ = float(np.quantile(data, self.threshold_quantile))
        peaks = get_peaks(data, self.threshold_, min_gap)
        if len(peaks) < 10:
            raise ValueError(
                f"Only {len(peaks)} independent peaks found — "
                "lower threshold_quantile or shorten min_gap_days."
            )
 
        self.shape_, self.scale_ = fit_gpd(peaks, self.threshold_)
        self.zeta_ = len(peaks) / len(data)
        N = int(self.return_period_years * self.samples_per_year)
        self.return_level_ = return_level(
            self.threshold_, self.shape_, self.scale_, self.zeta_, N
        )
 
        self.n_peaks_ = len(peaks)
        self.peaks_per_year_ = len(peaks) / (len(data) / self.samples_per_year)
        return self
 
    def transform(self, data):
        if self.return_level_ is None:
            raise RuntimeError("Call .fit() before .transform().")
        return np.asarray(data) / self.return_level_
 
    def inverse_transform(self, data):
        if self.return_level_ is None:
            raise RuntimeError("Call .fit() before .inverse_transform().")
        return np.asarray(data) * self.return_level_
 
    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    def summary(self):
        return (
            f"RiverStageNormalizer fit:\n"
            f"  threshold (q={self.threshold_quantile}): {self.threshold_:.3f}\n"
            f"  independent peaks:                  {self.n_peaks_} "
            f"({self.peaks_per_year_:.1f}/yr)\n"
            f"  GPD shape (xi):                     {self.shape_:+.3f}\n"
            f"  GPD scale (sigma):                  {self.scale_:.3f}\n"
            f"  {self.return_period_years}-yr return level (normalizer): "
            f"{self.return_level_:.3f}"
        )