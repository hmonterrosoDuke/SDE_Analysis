import numpy as np
import jumpdiff as jd


import numpy as np
import jumpdiff as jd


def compute_KM(site_dict, bw=0.3, bins=100, power=4, dt=15):
    """
    Compute Kramers-Moyal coefficients (with jump correction) for one site.
    Adds 'KM_results' in-place to site_dict.
    """
    y = site_dict['y_norm']
    if y is None:
        site_dict['KM_results'] = None
        return

    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if len(y) < 1000:
        site_dict['KM_results'] = None
        return

    edges, moments = jd.moments(y, bw=bw, power=power, bins=bins, norm=True)

    norm_factor = 1.0 / dt

    F1 = moments[1] * norm_factor
    F2 = moments[2] * norm_factor
    F3 = moments[3] * norm_factor
    F4 = moments[4] * norm_factor

    # Jump correction: D2_diffusive = F2 - F3^2 / F4
    # (Anvari/Tabar correction — removes jump contribution from second moment)
    jump_contribution = np.where(np.abs(F4) > 1e-10, F3**2 / F4, 0.0)
    F2_corrected = F2 - jump_contribution

    site_dict['KM_results'] = {
        'edges': edges,
        'F1': F1,
        'F2': F2,
        'F2_corrected': F2_corrected,
        'F3': F3,
        'F4': F4,
    }


