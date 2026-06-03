import numpy as np

def _clip_region(x, I, x_min=None, x_max=None):
    mask = np.ones_like(x, dtype=bool)
    
    if x_min is not None:
        mask &= (x >= x_min)
        
    if x_max is not None:
        mask  &= (x <= x_max)
        
    return x[mask], I[mask]

def analysis_window(x, center, width):
    return center - width / 2, center + width / 2

def centroid(x, I, x_min=None, x_max=None):
    """
    Intensity-weighted centroid.

    Returns
    -------
    float
    """
    x, I = _clip_region(x=x, I=I, x_min=x_min, x_max=x_max)
    I_sum = np.sum(I)
    
    if (I_sum) <= 0:
        return np.nan
    return np.sum(x * I) / I_sum

def rms_width(x, I, x_min=None, x_max=None):
    """
    Intensity-weighted centroid.

    Returns
    -------
    float
    """
    x, I = _clip_region(x=x, I=I, x_min=x_min, x_max=x_max)
    x0 = centroid(x, I)
    
    I_sum = np.sum(I)
    
    if I_sum <= 0:
        return np.nan
    
    variance = np.sum(I * (x - x0)**2) / I_sum
    return 2 * np.sqrt(variance)

def fwhm(x, I, x_min=None, x_max=None):
    """
    Full width at half mean

    Returns
    -------
    FWHM : float
    """
    x, I = _clip_region(x=x, I=I, x_min=x_min, x_max=x_max)
    if np.max(I) <= 0:
        return np.nan
    
    half = 0.5 * np.max(I)
    
    above = np.where(I >= half)[0]
    if len(above) < 2:
        return np.nan
    return x[above[-1]] - x[above[0]]