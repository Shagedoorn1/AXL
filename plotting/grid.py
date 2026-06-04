import numpy as np

class Grid:
    """
    Minimal structured grid container.

    Convention:
    - field shape is (Nz, Nx)
    - z = rows, x = columns
    """
    def __init__(self, x, z):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        
        if len(x) < 2 or len(z) < 2:
            raise ValueError("Grid too small")
        
        if not np.allclose(np.diff(z), np.diff(z)[0]):
            raise ValueError("z must be uniform")
        
        self.x = x
        self.z = z
        
        self.dx = x[1] - x[0]
        self.dz = z[1] - z[0]
        
        self.Nx = len(x)
        self.Nz = len(z)
        
    @property
    def extent(self):
        return [self.x[0], self.x[-1], self.z[0], self.z[-1]]
    
    @property
    def shape(self):
        return (self.Nz, self.Nx)
    
    def crop_x(self, field, x_min, x_max):
        mask = (self.x >= x_min) & (self.x <= x_max)
        return self.x[mask], field[:, mask]
    
    def crop_z(self, field, z_min, z_max):
        mask = (self.z >= z_min) & (self.z <= z_max)
        return self.z[mask], field[mask, :]