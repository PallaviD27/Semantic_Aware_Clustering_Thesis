import math
import torch


# def _quantizer_step(bits, upper_bound):
#     if type(bits) is not int or not 1 <= bits <= 16:
#         raise ValueError("bits must be an integer between 1 and 16.")
#
#     if not math.isfinite(upper_bound) or upper_bound <= 0:
#         raise ValueError("upper_bound must be finite and positive.")
#
#     return upper_bound / (2**bits - 1)
def _quantizer_step(bits, upper_bound, lower_bound=0.0):
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer between 1 and 16.")

    if (
        not math.isfinite(lower_bound)
        or not math.isfinite(upper_bound)
        or lower_bound >= upper_bound
    ):
        raise ValueError("Bounds must be finite, with lower_bound < upper_bound.")

    return (upper_bound - lower_bound) / (2**bits - 1)



# def quantize_features(x, bits=8, upper_bound=4.0):
#     """Transmitter: convert features into integer quantization indices."""
#     step = _quantizer_step(bits, upper_bound)
#
#     x_clipped = torch.clamp(x, min=0.0, max=upper_bound)
#
#     indices = torch.round(x_clipped / step)
#     indices = indices.clamp(0, 2**bits - 1).to(torch.int64)
#
#     return indices

def quantize_features(x, bits=8, upper_bound=4.0, lower_bound=0.0):
    """Transmitter: convert signed features into nonnegative indices."""
    step = _quantizer_step(bits, upper_bound, lower_bound)

    x_clipped = torch.clamp(x, min=lower_bound, max=upper_bound)

    indices = torch.round((x_clipped - lower_bound) / step)
    indices = indices.clamp(0, 2**bits - 1).to(torch.int64)

    return indices


# def dequantize_features(
#     received_indices,
#     bits=8,
#     upper_bound=4.0,
#     *,
#     dtype=torch.float32,
#     ste_reference=None,
# ):
#     """Receiver: reconstruct features from successfully received indices."""
#     step = _quantizer_step(bits, upper_bound)
#
#     x_reconstructed = received_indices.to(dtype=dtype) * step
#
#     # Training-only surrogate gradient through quantization.
#     if ste_reference is not None:
#         x_clipped = torch.clamp(
#             ste_reference, min=0.0, max=upper_bound
#         )
#         x_reconstructed = (
#             x_reconstructed.detach()
#             + (x_clipped - x_clipped.detach())
#         )
#
#     return x_reconstructed

def dequantize_features(
    received_indices,
    bits=8,
    upper_bound=4.0,
    *,
    lower_bound=0.0,
    dtype=torch.float32,
    ste_reference=None,
):
    """Receiver: reconstruct signed features from received indices."""
    step = _quantizer_step(bits, upper_bound, lower_bound)

    x_reconstructed = (
        received_indices.to(dtype=dtype) * step + lower_bound
    )

    # Training-only surrogate gradient through quantization.
    if ste_reference is not None:
        x_clipped = torch.clamp(
            ste_reference, min=lower_bound, max=upper_bound
        )
        x_reconstructed = (
            x_reconstructed.detach()
            + (x_clipped - x_clipped.detach())
        )

    return x_reconstructed

def index_to_bits(x_quantized, bits=8):
    """Converts quantized indices to bits."""
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer between 1 and 16.")
    if x_quantized.dtype !=torch.int64:
        raise TypeError("x_quantized must be torch.int64.")
    if torch.any((x_quantized < 0)|(x_quantized >= 2**bits)):
        raise ValueError("x_quantized must be between 0 and 2**bits-1, inclusive")

    shifts = torch.arange(bits-1,-1,-1, device=x_quantized.device)

    x_quantized = x_quantized.unsqueeze(-1)

    bit_rep = (x_quantized >> shifts) & 1

    return bit_rep


def bits_to_index(bits_received, bits=8):
    """Converts bits into indices."""
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer between 1 and 16.")

    if bits_received.ndim ==0 or bits_received.shape[-1] != bits:
        raise ValueError("The last dimension must equal bits")

    if torch.any((bits_received !=0) & (bits_received !=1)):
        raise ValueError("bits_received must contain only 0 or 1")

    shifts = torch.arange(bits-1,-1,-1, device=bits_received.device)

    weights = 2**shifts

    indices = (bits_received.to(torch.int64) * weights).sum(dim=-1)

    return indices
