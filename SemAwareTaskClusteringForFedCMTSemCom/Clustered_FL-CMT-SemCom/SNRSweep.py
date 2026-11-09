import numpy as np
import math


def sigma_sweep(Pref,h,SNRb_db):
    noise_variance = (h**2 * Pref)/(10**(SNRb_db/10))
    sigma_sweep = math.sqrt(noise_variance)
    return sigma_sweep