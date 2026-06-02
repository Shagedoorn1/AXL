import numpy as np

def lens_xu_single(x, n_lens=1.53, n_out=1.0, r_A=10.0, rA_over_a=0.7, h0=70.0):
    a = r_A / rA_over_a
    xi = n_lens / np.sqrt(n_lens**2 - n_out**2)
    e = np.sqrt(1.0 - 1.0 / xi**2)
    b = xi * a

    term = np.maximum(0.0, 1.0 - (x / a)**2)
    h = b * np.sqrt(term)
    h -= np.max(h)
    h += h0
    h[np.abs(x) > r_A] = 0.0
    return np.maximum(h, 0.0)


def lens_xu_triplet(x, n_lens=1.53, r_A=10.0, rA_over_a=0.7, h0=70.0):
    
    return lens_xu_array(x, n_lenses=3, n_lens=n_lens, r_A=r_A, rA_over_a=rA_over_a, h0=h0)

def lens_xu_array(x, n_lenses=3, n_lens=1.53, r_A=10.0, rA_over_a=0.7, h0=70):
    spacing = 2.0 * r_A
    
    centers = (
        np.arange(n_lenses) - (n_lenses - 1) / 2
    )
    
    profiles = [
        lens_xu_single(
            x - c,
            n_lens=n_lens,
            r_A=r_A,
            rA_over_a=rA_over_a,
            h0=h0,
        )
        for c in centers
    ]

    return np.maximum.reduce(profiles)