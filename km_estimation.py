import numpy as np
import jumpdiff as jd


def compute_KM(site_dict, bw=0.3, bins=100, power=4, dt=15):
    
    y = site_dict['df']['y_norm'].to_numpy()
    y = y[np.isfinite(y)]
    timescale = site_dict['timescale']
    
    edges, moments = jd.moments(y, bw=bw, power=power, bins=bins, norm=True)
    
    
    norm_factor = 1 / dt

    F2 = moments[2]* norm_factor
    F3 = moments[3]* norm_factor
    F4 = moments[4]* norm_factor

    # Jump correction: remove lambda(y)*xi(y)^2 = F3^2/F4
    jump_contribution = np.where(np.abs(F4) > 1e-10, F3**2 / F4, 0)
    F2_corrected = F2 - jump_contribution

    km_results = {
        'edges': edges,
        'F1': moments[1] * norm_factor,
        'F2': F2,
        'F2_corrected': F2_corrected,
        'F3': F3,
        'F4': F4,
    }

    site_dict['KM_results'] = km_results


