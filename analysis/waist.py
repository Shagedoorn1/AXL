from .spot import *
import numpy as np

def find_waist(x, z, field, z_min=50.0, z_max=700.0):
    """
    Search from the deepest z upward to find the first (tightest) waist.
    This avoids getting confused by the broad converging beam above the focus.
    """
    x = np.asarray(x)
    w0_local = np.full(len(z), np.nan)
    
    for j, zi in enumerate(z):
        if not (z_min < zi < z_max):
            continue
        w0, _, _ = spot_analysis(x, np.abs(field[j, :])**2)
        if not np.isnan(w0):
            w0_local[j] = w0
    
    valid = np.isfinite(w0_local)
    if not np.any(valid):
        return None
    
    # Search from bottom (deepest z) upward
    valid_idx = np.where(valid)[0]
    # Start from the deepest valid point, walk up to find minimum
    deepest = valid_idx[-1]
    shallowest = valid_idx[0]
    
    # Find minimum in the bottom half of the valid range (near substrate bottom)
    bottom_half = valid_idx[valid_idx >= np.median(valid_idx)]
    if len(bottom_half) == 0:
        bottom_half = valid_idx
    
    local_min = np.argmin(w0_local[bottom_half])
    waist_idx = bottom_half[local_min]
    
    return {
        'z_waist': z[waist_idx],
        'idx_waist': waist_idx,
        'w0_waist': w0_local[waist_idx],
        'w0_all': w0_local,
    }