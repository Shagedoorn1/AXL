import numpy as np
import warnings

from .waist import centroid
from .waist import rms_width
from .waist import fwhm

def waist_vs_z(x, z, I, method="rms", x_min=None, x_max=None):
    w = np.zeros(len(z))
    
    metric = {
        "rms": rms_width,
        "fwhm": fwhm
    }[method]
    
    for j in range(len(z)):
        w[j] = metric(x=x, I=I[j], x_min=x_min, x_max=x_max)
        
    return z, w

def centroid_vs_z(x, z, I, x_min=None, x_max=None):
    xc = np.zeros(len(z))
    
    for j in range(len(z)):
        xc[j] = centroid(x=x, I=I[j], x_min=x_min, x_max=x_max)
    
    return z, xc

def focus_position(x, z, I, method="rms", x_min=None, x_max=None):
    zvals, w = waist_vs_z(x=x, z=z, I=I, method=method, x_min=x_min, x_max=x_max)
    
    idx = np.nanargmin(w)
    
    if idx == 0 or idx == len(w) - 1:
        warnings.warn(f"Detected focus occurs at simulation boundary: {z[idx]}. Real focus might be outside simulation box")
    
    return zvals[idx], w[idx]

def focus_metrics(
    x,
    z,
    intensity,
    x_min=None,
    x_max=None,
):
    focus_z, waist = focus_position(x, z, intensity, method="rms", x_min=x_min, x_max=x_max)

    return {
        "focus_z": focus_z,
        "waist": waist,
    }