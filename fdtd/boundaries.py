"""
Boundary-condition utilities.

Future home for:
- Mur ABC
- PML
- CPML
- PEC / PMC boundaries
"""
from numba import njit, prange
from constants import c0
# -------------------------------------------------
# Boundaries
# -------------------------------------------------
@njit(cache=True, fastmath=True, parallel=True)
def apply_mur(Ey, Ey_prev, mur_x, mur_z, Nz, Nx):
    
    # -------------------------------------------------
    # Left / right boundaries
    # -------------------------------------------------
    
    for j in prange(1, Nz - 1):
        Ey[j, 0]      = (Ey_prev[j, 1]      + mur_x * (Ey[j, 1]     - Ey_prev[j, 0]     ))
        Ey[j, Nx - 1] = (Ey_prev[j, Nx - 2] + mur_x * (Ey[j, Nx -2] - Ey_prev[j, Nx - 1]))
        
        
    # -------------------------------------------------
    # Bottom / Top boundaries
    # -------------------------------------------------
    
    for i in prange(1, Nx - 1):
        Ey[0, i]      = (Ey_prev[1, i]      + mur_z * (Ey[1, i]      - Ey_prev[0, i]     ))
        Ey[Nz - 1, i] = (Ey_prev[Nz - 2, i] + mur_z * (Ey[Nz - 2, i] - Ey_prev[Nz - 1, i]))
        
    # -------------------------------------------------
    # Corners
    # -------------------------------------------------
    Ey[0, 0]       = 0.5 * (Ey[0, 1]       + Ey[1, 0]      )
    Ey[0, Nx-1]    = 0.5 * (Ey[0, Nx-2]    + Ey[1, Nx-1]   )
    Ey[Nz-1, 0]    = 0.5 * (Ey[Nz-2, 0]    + Ey[Nz-1, 1]   )
    Ey[Nz-1, Nx-1] = 0.5 * (Ey[Nz-2, Nx-1] + Ey[Nz-1, Nx-2])
    
class MurBoundary:
    def __init__(self, dt, dx, dz):
        self.mur_x = (c0 * dt - dx) / (c0 * dt + dx)
        self.mur_z = (c0 * dt - dz) / (c0 * dt + dz)
        
    def apply(self, Ey, Ey_prev, Nz, Nx):
        apply_mur(Ey=Ey, Ey_prev=Ey_prev, mur_x=self.mur_x, mur_z=self.mur_z, Nz=Nz, Nx=Nx)
    

# -------------------------------------------------
# Abstraction
# -------------------------------------------------
BOUNDARIES = {
    "mur": MurBoundary,
}

def get_boundary(name, **kwargs):
    if name not in BOUNDARIES:
        raise ValueError(f"Unknown boundary: {name}")
    return BOUNDARIES[name](
        kwargs["dt"],
        kwargs["dx"],
        kwargs["dz"],
    )