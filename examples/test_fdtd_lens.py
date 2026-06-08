from optics.maps import uniform, paint, paint_profile
from optics.lenses import lens_xu_single

from plotting.plots import plot_index_map, plot_field
from plotting.style import Style

from analysis.spot import focus_position

from fdtd.solver import FDTD
from fdtd.sources import get_source

from constants import c0

import numpy as np
import matplotlib.pyplot as plt
def main():
    r_A = 20
    h0 = 44.72135955#43.697
    
    dx = 0.05
    dz = 0.05
    x = np.arange(-50, 50 + dx, dx)
    z = np.arange(-60, 10 + dz, dz)
    h = lens_xu_single(x=x, n_lens=1.53, n_out=1.0, r_A=r_A, rA_over_a=1, h0=h0)
    
    n_map = paint_profile(uniform(x=x, z=z, n=1.0), z=z, bottom=0, profile=h, n=1.53)

    source = get_source(
        name="continuous_wave",
        x=x,
        src_j=5,
        w_in=40.0,
        amplitude=1.0,
        omega=2 * np.pi * c0 / 1.55
    )
    
    fdtd = FDTD(x=x, z=z, wavelength=1.55, n_map=n_map, boundary_type="mur", source=source)
    
    style = Style(dark=False, font_size=9, cmap="RdBu", grid=True, watermark="H.E.A.V.Y.")
    fig = plot_index_map(n_map=n_map, grid=fdtd.grid, style=style)
    plt.show()
    
    
    n_steps=5000
    fdtd.run(n_steps=n_steps)
    style = Style(dark=True, font_size=9, cmap="RdBu", grid=True, watermark="H.E.A.V.Y.")
    fig = plot_field(Ey=fdtd.Ey, grid=fdtd.grid, style=style, n_map=n_map)
    
    plt.show()
    
    style = Style(dark=False, font_size=9, cmap="inferno", grid=True, watermark="H.E.A.V.Y.")
    fig = plot_field(Ey=np.log10(fdtd.get_intensity())+1e-12, grid=fdtd.grid, style=style, title=r"$\log_{10}(I)$", n_map=n_map)
    
    plt.show()
    
    zvals, focus = focus_position(x, z, fdtd.get_intensity(), method="rms", x_min=-r_A/2, x_max=r_A/2)
    print(f"focus = {focus:.2f}")
    print(f"z_focus = {zvals:.2f}")
    
    print(f"src_j = {source.src_j}")
    print(f"z[src_j] = {z[source.src_j]}")
    print("max Ey =", np.max(np.abs(fdtd.Ey)))
    print("max intensity =", np.max(fdtd.intensity))
    print("n_accum =", fdtd.n_accum)
    print(f"min E = {np.min(fdtd.Ey)}")
    print(f"max E = {np.max(fdtd.Ey)}")
    print(f"min I = {np.min(fdtd.get_intensity())}")
    print(f"max I = {np.max(fdtd.get_intensity())}")
if __name__ == "__main__":
    main()