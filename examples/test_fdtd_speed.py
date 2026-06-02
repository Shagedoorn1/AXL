import time
import numpy as np
from fdtd.sources import get_source
from fdtd.solver import FDTD
from constants import *
from numba import get_num_threads

def make_free_space(nz, nx):
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
    # source
    # ----------------------------
    src_j = int(0.01 * Nz)
    source = get_source("continuous_wave", x=x, src_j=src_j, w_in=5.0, amplitude=5.0, omega=2*np.pi*c0/1.55)
    # ----------------------------
    # solver
    # ----------------------------
    fdtd = FDTD(x, z, wavelength=1.55, n_map=n_map, boundary_type="mur", source=source)
    # ----------------------------
    # warmup (IMPORTANT for numba JIT)
    # ----------------------------
    print("Warming up JIT...")
    
    for n in range(5000):
        fdtd._step(n, accumulate=False)

    fdtd.reset()

    # ----------------------------
    # benchmark
    # ----------------------------
    n_steps = 5000

    print(f"Running {n_steps} steps...")
    t0 = time.perf_counter()

    for n in range(n_steps):
        fdtd._step(n, accumulate = False)

    t1 = time.perf_counter()

    dt = t1 - t0

    print("\n--- RESULTS ---")
    print(f"Time total: {dt:.4f} s")
    print(f"Time per step: {dt / n_steps * 1e3:.6f} ms")
    print(f"Grid size: {Nz} x {Nx} = {Nz * Nx / 1e6:.3f} M cells")
    print(f"Cell updates per second: {(Nz * Nx * n_steps) / dt / 1e6:.3f} M/s")
    print(f"Kernel Troughput: {(2 * Nz * Nx * n_steps) / dt / 1e9:3f} B/s")

if __name__ == "__main__":
    print(get_num_threads())
    main()