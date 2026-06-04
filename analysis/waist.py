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
    
    peak = np.argmax(I)
    
    left = np.where(I[:peak] < half)[0]
    
    if len(left) == 0:
        return np.nan
    
    i1 = left[-1]
    i2 = i1 + 1
    
    x_left = np.interp(half, [I[i1], I[i2]], [x[i1], x[i2]])
    
    right = np.where(I[peak:] < half)[0]
    if len(right) == 0:
        return np.nan

    i2 = peak + right[0]
    i1 = i2 - 1

    x_right = np.interp(
        half,
        [I[i2], I[i1]],
        [x[i2], x[i1]]
    )

    return x_right - x_left