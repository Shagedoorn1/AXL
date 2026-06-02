"""
Test FDTD free-space propagation with MP4 export.
"""
import numpy as np
from fdtd.solver import FDTD


def make_free_space(nz, nx):
    """Return refractive index map for free space."""
    return np.ones((nz, nx), dtype=np.float64)


def main():
    # ----------------------------
    # grid
    # ----------------------------
    dx = 0.1
    dz = 0.1

    x = np.linspace(-20, 20, int(40 / dx) + 1)
    z = np.linspace(0, 80, int(80 / dz) + 1)

    Nz, Nx = len(z), len(x)

    # ----------------------------
    # material: free space
    # ----------------------------
    n_map = make_free_space(Nz, Nx)

    # ----------------------------
    # solver
    # ----------------------------
    wavelength = 1.55  # µm
    fdtd = FDTD(x, z, wavelength=wavelength, n_map=n_map)

    # Set continuous-wave source (Gaussian-profile plane wave)
    fdtd.set_source(
        source_type="continuous_wave",
        z_src=z[int(0.1 * Nz)],  # 10% into domain
        w_in=5.0,                 # Gaussian width (µm)
        amplitude=1.0,            # field amplitude
        omega=fdtd.omega           # angular frequency (computed from wavelength)
    )

    # ----------------------------
    # export to MP4
    # ----------------------------
    n_steps = 2000
    frame_skip = 20  # record every 20th step
    downsample = 2
    fps = 30

    print(f"Exporting {Nz}x{Nx} grid for {n_steps} steps to MP4...")
    print(f"  Frame skip: {frame_skip}")
    print(f"  Downsample: {downsample}x")
    print(f"  FPS: {fps}")

    fdtd.export_mp4(
        filename="fdtd_freespacejpropagation.mp4",
        n_steps=n_steps,
        frame_skip=frame_skip,
        downsample=downsample,
        fps=fps
    )

    print("Done!")


if __name__ == "__main__":
    main()
