import numpy as np

def uniform(x, z, n=1.0):
    return np.full((len(z), len(x)), n, dtype=float)

def paint(n_map, mask, n):
    n_map[mask] = n
    
def add_layer(n_map, z, z_min, z_max, n):
    mask = (z[:, None] >= z_min) & (z[:, None] <= z_max)
    n_map[mask] = n

def paint_profile(n_map, x, z, bottom, profile, n):
    Z = z[:, None]
    
    mask = (Z >= bottom) & (Z <= profile[None, :])
    
    n_map[mask] = n