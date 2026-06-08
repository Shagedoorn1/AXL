from .kernels import update_H, update_E, accumulate_intensity
from .boundaries import get_boundary
from plotting import Grid
from constants import c0, eps0, mu0
import numpy as np


class FDTD:
    """
    Core solver.

    Convention:
    - All physics lives in kernels
    - Boundary is an object
    - Source is an object (optional)
    - run() only orchestrates _step()
    """

    def __init__(self, x, z, wavelength, n_map, boundary_type="mur", source=None):

        # ----------------------------
        # grid checks
        # ----------------------------
        if len(x) < 2 or len(z) < 2:
            raise ValueError("Grid too small")

        if not np.allclose(np.diff(x), np.diff(x)[0]):
            raise ValueError("x must be uniform")

        if not np.allclose(np.diff(z), np.diff(z)[0]):
            raise ValueError("z must be uniform")

        if n_map.shape != (len(z), len(x)):
            raise ValueError("n_map shape mismatch")

        # ----------------------------
        # constants
        # ----------------------------
        self.wavelength = wavelength
        self.omega = 2 * np.pi * c0 / wavelength

        # ----------------------------
        # grid
        # ----------------------------
        self.grid = Grid(x, z)

        self.idx = 1.0 / self.grid.dx
        self.idz = 1.0 / self.grid.dz
        
        # ----------------------------
        # material
        # ----------------------------
        self.n_map = np.asarray(n_map)
        self.eps = eps0 * self.n_map**2

        # ----------------------------
        # time step
        # ----------------------------
        self.dt = 0.99 / (c0 * np.sqrt(1/self.grid.dx**2 + 1/self.grid.dz**2))
        self.period = 2 * np.pi / self.omega
        self.steps_per_period = max(1, int(self.period / self.dt))
        self.burn_in_steps = 20 * self.steps_per_period

        # ----------------------------
        # fields
        # ----------------------------
        self.Ey = np.zeros((self.grid.Nz, self.grid.Nx))
        self.Hx = np.zeros((self.grid.Nz, self.grid.Nx))
        self.Hz = np.zeros((self.grid.Nz, self.grid.Nx))

        self.Ey_prev = np.zeros_like(self.Ey)

        # ----------------------------
        # coefficients
        # ----------------------------
        self.Chx = self.dt / (mu0 * self.grid.dz)
        self.Chz = self.dt / (mu0 * self.grid.dx)
        self.Ce = self.dt / self.eps

        # ----------------------------
        # boundary (IMPORTANT FIX)
        # ----------------------------
        self.boundary_type = boundary_type
        self.boundary = get_boundary(
            boundary_type,
            dt=self.dt,
            dx=self.grid.dx,
            dz=self.grid.dz,
        )

        # ----------------------------
        # source (optional object)
        # ----------------------------
        self.source = source

        # ----------------------------
        # diagnostics
        # ----------------------------
        self.probe_history = []
        self.probe_j = None

        self.Ey_prev = np.zeros_like(self.Ey)
        self.intensity = np.zeros_like(self.Ey)
        self.dose = np.zeros_like(self.Ey)
        self.n_accum = 0
        
        self.recorder = None
        self.n_steps = 0

    # ============================================================
    # core step (single source of truth)
    # ============================================================
    def _step(self, step, accumulate):

        update_H(self.Ey, self.Hx, self.Hz, self.Chx, self.Chz, self.grid.Nz, self.grid.Nx)
        update_E(self.Ey, self.Hx, self.Hz, self.Ce, self.idx, self.idz, self.grid.Nz, self.grid.Nx)

        if self.source is not None:
            self.source.apply(self.Ey, step, self.dt)

        self.boundary.apply(self.Ey, self.Ey_prev, self.grid.Nz, self.grid.Nx)

        accumulate_intensity(self.intensity, self.dose, self.Ey, self.grid.Nx, self.grid.Nz, self.dt)
        self.n_accum += 1
        
        if self.recorder is not None:
            self.recorder.capture(self.Ey, step)

        # memory swap (fast aliasing alternative)
        self.Ey_prev[:, :] = self.Ey
        
    def reset(self):
        self.Ey.fill(0.0)
        self.Hx.fill(0.0)
        self.Hz.fill(0.0)

        self.intensity.fill(0.0)
        self.n_accum = 0

        self.Ey_prev.fill(0.0)

        self.probe_history.clear()

    # ============================================================
    def run(self, n_steps=1000, probe_z=None, accumulate = True):
        self.n_steps = n_steps

        if probe_z is not None:
            self.probe_j = np.argmin(np.abs(self.z - probe_z))
            
        if self.recorder is not None:
            self.recorder.start(self)

        progress_interval = max(1, n_steps // 20)
        for step in range(n_steps):
            self._step(step, accumulate)
            if step % progress_interval == 0:
                print(f"step {step}/{n_steps}; {(step/n_steps)*100:.2f}")
            
        # probe
        if self.probe_j is not None:
            self.probe_history.append(self.Ey[self.probe_j, self.grid.Nx // 2])
        if self.recorder is not None:
            self.recorder.finalize()

    # ============================================================
    def get_intensity(self):
        return self.intensity / max(self.n_accum, 1)
    
    def get_dose(self):
        return self.dose