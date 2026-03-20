import numpy as np
import jumpdiff as jd


def compute_KM(site_dict, bw=2, power=4, dt=15):
    
    y = site_dict['df']['y_norm'].to_numpy()
    y = y[np.isfinite(y)]
    timescale = site_dict['timescale']
    
    edges, moments = jd.moments(y, bw=bw, power=power, norm=True)
    
    # Keep only positive edges
    positive = edges.flatten() >= 0
    
    norm_factor = timescale / dt
    
    km_results = {
        'edges': edges.flatten()[positive],
        'F1': moments[1].flatten()[positive] * norm_factor,
        'F2': moments[2].flatten()[positive] * norm_factor,
        'F3': moments[3].flatten()[positive] * norm_factor,
        'F4': moments[4].flatten()[positive] * norm_factor,
    }

    site_dict['KM_results'] = km_results


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
    

