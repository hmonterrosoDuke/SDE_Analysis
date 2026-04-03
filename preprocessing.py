import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def compute_acf(y, max_lag=None):
    y = np.asarray(y, dtype=float)
    
    n = len(y)
    if max_lag is None:
        max_lag = n // 4
    
    f = np.fft.fft(y, n=2*n)
    acf_full = np.fft.ifft(f * np.conj(f)).real
    
    acf = acf_full[:max_lag] / acf_full[0]  # normalize by lag-0 variance
    
    return acf

def integral_timescale(acf, dt=1.0):
    """
    e-folding timescale = lag where ACF drops to 1/e ≈ 0.37.
    
    Parameters
    ----------
    acf : array of autocorrelation values (lag 0 = 1.0)
    dt  : time step
    
    Returns
    -------
    T   : e-folding timescale in units of dt
    idx : index where ACF <= 1/e
    """
    threshold = 1 / np.e  # ≈ 0.368
    idx_arr = np.where(acf <= threshold)[0]
    
    if len(idx_arr) == 0:
        raise ValueError("ACF never drops to 1/e — increase max_lag")
    
    idx = idx_arr[0]
    T = idx * dt
    
    return T, idx

def deseasonalize(y, timestamps, method='stl'):
    """
    Remove seasonal signal from stage data before KM estimation.
    
    method='climatology' : subtract median by day-of-year (fast, nonparametric)
    method='stl'         : STL decomposition, returns remainder only
    """
    if method == 'climatology':
        doy = timestamps.dayofyear  # DatetimeIndex → no .dt needed
        df_tmp = pd.Series(y.values, index=doy)
        seasonal = df_tmp.groupby(level=0).transform('median')
        seasonal.index = y.index
        return y - seasonal, seasonal

    elif method == 'stl':
        # STL needs regular index — resample to hourly if 15-min
        stl = STL(y, period=365*24*4, robust=True)  # 4 obs/hr * 24hr * 365
        res = stl.fit()
        return pd.Series(res.resid, index=y.index), \
               pd.Series(res.seasonal, index=y.index)

def preprocess_data(site_data, norm_pctle = 0.995, dt = 15, deseasonalize_method=None):
    y = site_data['00065']
    y_intrp = y.interpolate(method='time').dropna()
    y_intrp = y_intrp - np.median(y_intrp)

    seasonal_component = None
    if deseasonalize_method is not None:
        y_intrp, seasonal_component = deseasonalize(
            y_intrp, y_intrp.index, method=deseasonalize_method)

    y_norm = y_intrp/np.quantile(y_intrp,norm_pctle)
    acf = compute_acf(y_norm)
    y_norm = y_norm - np.min(y_norm)
    timescale, idx = integral_timescale(acf, dt = dt)

    return y_norm, acf, timescale, idx, seasonal_component


def create_site_dictionary(site_data, dt=15, omega=0.49, c=2, std_window=50, deseasonalize_method= None):
    y_norm, acf, timescale, idx, seasonal_component = preprocess_data(site_data, norm_pctle=0.995, dt=dt, deseasonalize_method=deseasonalize_method)
    dy = np.diff(y_norm)
    
    local_std = pd.Series(y_norm).rolling(std_window, center=True, min_periods=10).std().values
    local_std = local_std[:-1]
    # Threshold scales with local volatility and dt
    threshold = c * local_std * (dt ** omega)

    # Jump if increment exceeds local threshold
    # TBD if I should keep this. 
    is_jump = dy > threshold
    
    df = pd.DataFrame({
        "y_norm": y_norm[:-1],
        "dy": dy,
        "local_std": local_std,
        "threshold": threshold,
        "is_jump": is_jump
    })

    site_dictionary = {'df': df,
                       'acf': acf,
                       'timescale': timescale,
                       'timescale_idx': idx,
                       'seasonal': seasonal_component
    }
    
    return site_dictionary


