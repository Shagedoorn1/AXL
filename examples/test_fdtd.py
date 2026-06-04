import numpy as np
import matplotlib.pyplot as plt

from constants import c0

from fdtd.solver import FDTD
from fdtd.sources import get_source

from optics.maps import uniform, paint

from plotting.plots import plot_field
from plotting.style import Style


def main():
    # Grid
    dx = 0.01
    dz = 0.01

    x = np.arange(-40, 40 + dx, dx)
    z = np.arange(0, 80+dz, dz)

    # Material
    n_map = uniform(x=x, z=z, n=1.0)

    # Source
    source = get_source(
        name="continuous_wave",
        x=x,
        src_j=5,
        w_in=20.0,
        amplitude=1.0,
        omega=2 * np.pi * c0 / 1.55
    )

    # Solver
    fdtd = FDTD(
        x=x,
        z=z,
        wavelength=1.55,
        n_map=n_map,
        source=source
    )

    # Run
    n_steps=10000
    fdtd.run(n_steps=n_steps)

    # Plot
    style = Style(dark=True, font_size=9, cmap="viridis", grid=True, watermark="FDTD@TUD")
    fig, ax = plot_field(
        Ey=fdtd.Ey,
        grid=fdtd.grid,
        style=style
    )

    ax.set_title(f"Free-space propagation, t = {fdtd.dt * n_steps:.2f} (fs)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")

    plt.show()

if __name__ == "__main__":
    main()