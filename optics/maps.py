import numpy as np

def uniform(x, z, n=1.0):
    return np.full((len(z), len(x)), n, dtype=float)

def paint(n_map, mask, n):
    if mask.shape != n_map.shape:
        mask = np.broadcast_to(mask, n_map.shape)
    n_map[mask] = n
    return n_map
    
def add_layer(n_map, z, z_min, z_max, n):
    mask = (z[:, None] >= z_min) & (z[:, None] <= z_max)
    mask = np.broadcast_to(mask, n_map.shape)
    n_map[mask] = n
    return n_map

def paint_profile(n_map, z, bottom, profile, n):
    """
    Paint material between

        bottom <= z <= profile(x)

    Parameters
    ----------
    profile : ndarray
        Absolute z coordinate of the upper surface.
    """
    Z = z[:, None]
    
    mask = (Z <= 0) & (Z >= -profile[None, :])
    
    n_map[mask] = n
    return n_map