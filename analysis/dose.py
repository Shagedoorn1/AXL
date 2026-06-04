import numpy as np

def peak_dose(dose):
    return np.max(dose)

def cured_area(mask, dx, dz):
    return np.sum(mask) * dx * dz

def cured_profile(x, z, dose, D_c):
    mask = cured_mask(dose, D_c)
    profile = np.full(len(x), np.nan)
    
    for i in range(len(x)):
        rows = np.where(mask[:, i])[0]
        
        if len(rows):
            profile[i] = z[rows.max()]
    return profile

def cured_mask(dose, D_c):
    return dose >= D_c

def jacobs_depth(dose, D_p, D_c):
    depth = np.zeros_like(dose)
    mask = dose > D_c
    
    depth[mask] = D_p * np.log(dose[mask] / D_c)
    
    return depth

