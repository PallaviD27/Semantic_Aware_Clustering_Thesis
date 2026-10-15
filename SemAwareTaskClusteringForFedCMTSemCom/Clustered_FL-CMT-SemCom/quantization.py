import torch


def quantize_features(x, bits=8, upper_bound=4.0, use_ste=True):
    """Quantize nonnegative features over [0, upper_bound]."""

    if not isinstance(bits, int) or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer between 1 and 16.")

    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive.")

    # Number of possible reconstruction values.
    levels = 2 ** bits

    # Distance between adjacent reconstruction values.
    step = upper_bound / (levels - 1)

    # Keep features within the range represented by the quantizer.
    x_clipped = torch.clamp(x, min=0.0, max=upper_bound)

    # Assign each feature an integer index.
    indices = torch.round(x_clipped / step)
    indices = indices.clamp(0, levels - 1).to(torch.int64)

    # Reconstruct feature values from their indices.
    x_reconstructed = indices.to(x.dtype) * step

    if use_ste:
        x_reconstructed = x_reconstructed.detach() + (x_clipped - x_clipped.detach())

    return x_reconstructed, indices