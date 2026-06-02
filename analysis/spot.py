import numpy as np

def spot_analysis(x, I, threshold=0.5):
    """
    Analyze a single intensity profile I(x).
    Returns (w0, x_c, fwhm).
    """
    x = np.asarray(x)
    I = np.asarray(I)
    
    # GUARD: ensure real, non-negative intensity
    if np.iscomplexobj(I):
        I = np.abs(I)**2
    
    I = np.real(I)
    if np.max(I) <= 0:
        return np.nan, np.nan, np.nan

    # Find peak nearest x=0
    center_idx = np.argmin(np.abs(x))
    peak_idx = center_idx
    while peak_idx > 0 and I[peak_idx-1] > I[peak_idx]:
        peak_idx -= 1
    while peak_idx < len(I)-1 and I[peak_idx+1] > I[peak_idx]:
        peak_idx += 1

    I_peak = I[peak_idx]
    halfmax = threshold * I_peak

    left = peak_idx
    while left > 0 and I[left] > halfmax:
        left -= 1
    right = peak_idx
    while right < len(I)-1 and I[right] > halfmax:
        right += 1

    region = slice(left, right+1)
    xw, Iw = x[region], I[region]
    if len(xw) < 3:
        return np.nan, np.nan, np.nan

    tot = np.trapezoid(Iw, xw)
    x_c = np.trapezoid(xw * Iw, xw) / tot
    var = np.trapezoid((xw - x_c)**2 * Iw, xw) / tot
    w0 = 2.0 * np.sqrt(var)
    fwhm = x[right] - x[left]
    return w0, x_c, fwhm
