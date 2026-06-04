import numpy as np
import matplotlib.pyplot as plt
from analysis.waist import centroid, rms_width, fwhm, analysis_window
from analysis.spot import waist_vs_z, focus_position, centroid_vs_z

def main():
    x = np.linspace(-50, 50, 2000)
    z = np.linspace(-50, 50, 2000)
    I0 = np.exp(-2 * x**2 / 5.0**2)
    I = np.tile(I0, (len(z), 1))
    
    zvals, waist = waist_vs_z(x, z, I)
    print(waist)
    z_focus, focus = focus_position(x, z, I, "rms")
    print(f"z_focus = {z_focus}")
    print(f"focus = {focus}")
    
    
    zf = 35.0
    
    I = np.zeros((len(z), len(x)))
    
    for j, zj in enumerate(z):
        w = 2.0 + 0.01 * (zj - zf)**2
        
        I[j] = np.exp(-2 * x**2 / w**2)
          
    z_focus, focus = focus_position(x, z, I, "rms")
    print("--- focussed gaussian ---")
    print(f"z_focus = {z_focus}")
    print(f"focus = {focus}")
    
    for j, zj in enumerate(z):

        xc = 0.2 * zj

        I[j] = np.exp(
            -2 * (x - xc)**2 / 4**2
        )
        
    z_vals, xc = centroid_vs_z(x, z, I)
    
    expected = 0.2 * z

    print(f"wandering beam error: {np.max(np.abs(xc - expected))}")
    
if __name__ == "__main__":
    main()