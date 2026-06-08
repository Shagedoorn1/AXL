from optics.maps import uniform, paint, paint_profile
from optics.lenses import lens_xu_single

from plotting.plots import plot_index_map, plot_field
from plotting.style import Style

from analysis.spot import focus_position

from fdtd.solver import FDTD
from fdtd.sources import get_source
from fdtd.recorder import MemmapRecorder
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
    
    n_map = paint_profile(uniform(x=x, z=z, n=1.0), z=z, bottom=0, profile=h, n=1.5)

    source = get_source(
        name="continuous_wave",
        x=x,
        src_j=5,
        w_in=40.0,
        amplitude=1.0,
        omega=2 * np.pi * c0 / 1.55
    )
    
    fdtd = FDTD(x=x, z=z, wavelength=1.55, n_map=n_map, boundary_type="mur", source=source)
    
    recorder = MemmapRecorder(
        stride = 10,
        filename="xu_lens.dat"
    )
    
    fdtd.recorder = recorder
    n_steps=5000
    fdtd.run(n_steps=n_steps)
    
    style = Style(dark=False, font_size=9, cmap="RdBu", grid=True, watermark="H.E.A.V.Y.")
    fig = plot_field(Ey=fdtd.Ey, grid=fdtd.grid, style=style)
    
    plt.show()
    
    frames = np.memmap(
        "xu_lens.dat",
        dtype=np.float32,
        mode="r",
        shape=(recorder.n_frames, fdtd.grid.Nz, fdtd.grid.Nx)
    )
    
    print("frame_idx =", recorder.frame_idx)
    print("max(frames) =", np.max(frames))
    print("min(frames) =", np.min(frames))

    recorder.render_frames(frames=frames, n_map=n_map, fps=30)
        
    
if __name__ == "__main__":
    main()