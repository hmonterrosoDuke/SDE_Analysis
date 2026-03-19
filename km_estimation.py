import numpy as np
import jumpdiff as jd


def compute_KM(site_dict, bw=0.5, power=3, dt=15):
    
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