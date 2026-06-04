import numpy as np

from analysis.dose import peak_dose, cured_mask, jacobs_depth, cured_profile

def main():
    dose = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    print(f"peak dose = {peak_dose(dose)}")
    
    dose = np.array([
        [1, 5, 8],
        [2, 7, 3]
    ])
    
    D_c = 4
    
    mask = cured_mask(dose, D_c)
    
    print(f"mask = {mask}")
    
    D_p = 10
    D_c = 100
    
    dose = np.array([
        [100],
        [100 * np.e],
        [100 * np.e**2]
    ])
    
    jac = jacobs_depth(dose, D_p, D_c)
    print(f"cure_depth = {jac.ravel()}")


    x = np.array([-1, 0, 1])

    z = np.array([0, 1, 2, 3])

    dose = np.array([
        [0, 0, 0],
        [5, 5, 0],
        [5, 5, 0],
        [5, 0, 0]
    ])

    D_c = 1
    
    profile = cured_profile(x, z, dose, D_c)
    print(f"profile = {profile}")
if __name__ == "__main__":
    main()