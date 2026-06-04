import numpy as np
import matplotlib.pyplot as plt
from analysis.waist import centroid, rms_width, fwhm, analysis_window

def main():
    x0 = 2.5
    w0 = 5.0
    
    x = np.linspace(-40, 40, 1000)
    
    I = np.exp(-2 * (x - x0)**2 / w0**2)
    
    xc_meas = centroid(x, I)
    w_meas = rms_width(x, I)
    f_meas = fwhm(x, I)
    
    xc_true = x0
    w_true  = w0
    f_true  = w0 * np.sqrt(2 * np.log(2))

    print("\n--- Analysis test ---")
    
    print(f"Centroid:")
    print(f"  measured = {xc_meas:.6f}")
    print(f"  expected = {xc_true:.6f}")
    print(f"  error    = {abs(xc_meas-xc_true):.3e}")
    
    print()

    print(f"RMS width:")
    print(f"  measured = {w_meas:.6f}")
    print(f"  expected = {w_true:.6f}")
    print(f"  error    = {abs(w_meas-w_true):.3e}")

    print()

    print(f"FWHM:")
    print(f"  measured = {f_meas:.6f}")
    print(f"  expected = {f_true:.6f}")
    print(f"  error    = {abs(f_meas-f_true):.3e}")
    
    x0 = 0.0
    w0 = 5
    
    x = np.linspace(-40, 40, 10000)
    
    I = np.exp(-2 * (x - x0)**2 / w0**2)
    
    xc_meas = centroid(x, I, x_min=-8, x_max=8)
    w_meas = rms_width(x, I, x_min=-8, x_max=8)
    f_meas = fwhm(x, I, x_min=-8, x_max=8)
    
    xc_true = x0
    w_true  = w0
    f_true  = w0 * np.sqrt(2 * np.log(2))

    print("\n--- Analysis test (clipped) ---")
    
    print(f"Centroid:")
    print(f"  measured = {xc_meas:.6f}")
    print(f"  expected = {xc_true:.6f}")
    print(f"  error    = {abs(xc_meas-xc_true):.3e}")
    
    print()

    print(f"RMS width:")
    print(f"  measured = {w_meas:.6f}")
    print(f"  expected = {w_true:.6f}")
    print(f"  error    = {abs(w_meas-w_true):.3e}")

    print()

    print(f"FWHM:")
    print(f"  measured = {f_meas:.6f}")
    print(f"  expected = {f_true:.6f}")
    print(f"  error    = {abs(f_meas-f_true):.3e}")
    
    
    rng = np.random.default_rng(5)
    x = np.linspace(-40, 40, 2000)
    
    I = np.exp(-2 * (x / 2.0)**2)
    
    I += 0.35 * np.exp(-2 * ((x - 25)/6)**2)
    I += 0.35 * np.exp(-2 * ((x + 25)/6)**2)
    
    I += 0.02 * rng.random(len(x))

    I = np.maximum(I, 0.0)
    
    print("No clipping")
    print("centroid =", centroid(x, I))
    print("rms      =", rms_width(x, I))
    print("fwhm     =", fwhm(x, I))
    
    xmin, xmax = analysis_window(
        x,
        center=0.0,
        width=10.0
    )

    print("Clipped")
    print("centroid =", centroid(x, I, xmin, xmax))
    print("rms      =", rms_width(x, I, xmin, xmax))
    print("fwhm     =", fwhm(x, I, xmin, xmax))
    
    plt.figure()

    plt.plot(x, I)

    plt.axvline(xmin, ls="--")
    plt.axvline(xmax, ls="--")

    plt.title("Analysis window test")
    plt.show()
    
if __name__ == "__main__":
    main()