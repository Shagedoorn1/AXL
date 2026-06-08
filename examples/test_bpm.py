import numpy as np
import matplotlib.pyplot as plt

from optics.lenses import lens_xu_single
from optics.maps import uniform, paint_profile

from analysis.spot import focus_position

from bpm.solver import BPM
from plotting import plot_field

def gaussian(x, w0=10.0):
    return np.exp(-(x**2) / (w0**2)).astype(np.complex128)

def main():
    r_A = 20
    h0 = 44.72135955#43.697
    
    dx = 0.05
    dz = 0.05
    x = np.arange(-50, 50 + dx, dx)
    z = np.arange(-60, 10 + dz, dz)
    h = lens_xu_single(x=x, n_lens=1.53, n_out=1.0, r_A=r_A, rA_over_a=1, h0=h0)
    
    n_map = paint_profile(uniform(x=x, z=z, n=1.0), z=z, bottom=0, profile=h, n=1.53)
    #n_map = np.ones((len(z),len(x)))
    bpm = BPM(x=x, z=z, wavelength=1.55, n_map=n_map, n0=1.0)
    
    bpm.set_input(field=gaussian(x=x, w0=40.0))
    
    bpm.run()
    
    field = bpm.field
    intensity = np.abs(field)**2
    
    print(np.min(np.angle(bpm.D)))
    print(np.max(np.angle(bpm.D)))
    
    plt.figure(figsize=(7, 4))

    plt.plot(x, intensity)

    plt.title("BPM final field intensity")
    plt.xlabel("x [µm]")
    plt.ylabel("|E|²")

    plt.tight_layout()
    plt.show()
    
    I = np.abs(bpm.Ey)**2
    
    
    plot_field(Ey=np.log10(I)+1e-12, grid=bpm.grid, style=None, n_map=n_map)
    plt.show()
    
    zvals, focus = focus_position(x, z, I, method="rms", x_min=-r_A/2, x_max=r_A/2)
    print(f"focus = {focus:.2f}")
    print(f"z_focus = {zvals:.2f}")
    
    plt.plot(x, np.abs(bpm.Ey[0])**2, label="input")
    plt.plot(x, np.abs(bpm.Ey[-1])**2, label="output")
    plt.legend()
    plt.show()
    
    
    I0 = np.abs(bpm.Ey[0])**2
    I1 = np.abs(bpm.Ey[-1])**2
    P0 = np.trapezoid(I0, x)
    P1 = np.trapezoid(I1, x)

    print(P0)
    print(P1)

    print("input width")
    print(np.sqrt(np.sum(x**2 * I0)/np.sum(I0)))

    print("output width")
    print(np.sqrt(np.sum(x**2 * I1)/np.sum(I1)))
    
if __name__ == "__main__":
    main()