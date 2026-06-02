from numba import njit, prange

@njit(cache=True, fastmath=True, parallel=True)
def update_H(Ey, Hx, Hz, Chx, Chz, Nz, Nx):
    
    for j in prange(Nz - 1):
        Ey_j = Ey[j]
        Ey_j1 = Ey[j + 1]
        
        Hx_j = Hx[j]
        Hz_j = Hz[j]
        
        for i in range(Nx - 1):
            dE_z = Ey_j1[i] - Ey_j[i]
            dE_x = Ey_j[i + 1] - Ey_j[i]
            
            Hx_j[i] += Chx * dE_z
            Hz_j[i] -= Chz * dE_x
    
@njit(cache=True, fastmath=True, parallel=True)
def update_E(Ey, Hx, Hz, Ce, idx, idz, Nz, Nx):

    for j in prange(1, Nz - 1):
        Hx_j  = Hx[j]
        Hx_jm = Hx[j - 1]

        Hz_j  = Hz[j]

        Ey_j  = Ey[j]
        Ce_j  = Ce[j]
        for i in range(1, Nx - 1):
           curl_h = ((Hx_j[i] - Hx_jm[i]) * idz - (Hz_j[i] - Hz_j[i - 1]) * idx)
           Ey_j[i] += Ce_j[i] * curl_h

@njit(cache=True, fastmath=True, parallel=True)
def accumulate_intensity(intensity, Ey, Nx, Nz):
    
    for j in prange(Nz):
        Ey_j = Ey[j]
        I_j = intensity[j]
        for i in range(Nx):
            e = Ey_j[i]
            I_j[i] += e * e
            
@njit(cache=True, fastmath=True)
def inject_source(Ey, src_profile, src_j, source_value, Nx):
    Ey_row = Ey[src_j]
    
    for i in prange(Nx):
        Ey_row[i] += source_value * src_profile[i]
