# This file contains functions for modulation and demodulation
import torch


def modulation(bit_rep):
    '''Does BPSK modulation.
    Takes quantized binary input and maps 0 to -1 and 1 to +1'''

    x_modulated = 2*bit_rep.to(torch.float32) - 1

    return x_modulated

def demodulation(x_modulated):
    '''Does BPSK demodulation.'''
    bits_received = (x_modulated >= 0).to(torch.int64)
    return bits_received