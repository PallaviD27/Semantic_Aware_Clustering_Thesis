import math

import torch
import numpy as np

def snr_db_to_sigma_weight_no_normalize(w, snr_db_agg=None):
    if snr_db_agg is None:
        return None

    snr_linear_weight_no_normalize = 10 ** (snr_db_agg/10.0)
    weight_power_no_normalize = w.pow(2).mean()
    noise_power_no_normalize = weight_power_no_normalize / snr_linear_weight_no_normalize
    sigma_agg = torch.sqrt(noise_power_no_normalize)
    return sigma_agg

def awgn_agg_no_normalize(w, sigma_agg=None, snr_db_agg=None):
    if snr_db_agg is not None:
        sigma_agg = snr_db_to_sigma_weight_no_normalize(w,snr_db_agg)

    if sigma_agg is None:
        return w

    noise_weights = torch.randn_like(w, device=w.device) * sigma_agg
    return w + noise_weights



# Functions to apply AWGN to model updates when normalization happens
def rms_power_normalize_weight(w, eps=1e-8):
    weight_power = w.pow(2).mean()
    return w / torch.sqrt(weight_power + eps)

def snr_db_to_sigma_weight_normalize(snr_db_agg):
    snr_linear_weight_normalize = 10 ** (snr_db_agg/10.0)
    sigma_agg = math.sqrt(1/snr_linear_weight_normalize)
    return sigma_agg

def awgn_agg_normalize(w, sigma_agg=None, snr_db_agg=None):
    if snr_db_agg is not None:
        sigma_agg = snr_db_to_sigma_weight_normalize(snr_db_agg)

    if sigma_agg is None:
        return w

    noise_weights = torch.randn_like(w,device=w.device) * sigma_agg
    return w + noise_weights






