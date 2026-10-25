import math

import torch


def normalize_padding_mask(features, mask, visual=False):
    if features is None:
        if mask is not None:
            raise ValueError("padding mask supplied for a missing view")
        return None
    shape = features.shape[:-1]
    if mask is None:
        return torch.zeros(shape, device=features.device, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("padding masks must be boolean (True means missing)")
    mask = mask.to(features.device)
    if visual and tuple(mask.shape) == tuple(shape[:2]):
        mask = mask[:, :, None, None].expand(shape)
    elif visual and tuple(mask.shape) == (shape[0], math.prod(shape[1:])):
        mask = mask.reshape(shape)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"padding mask shape {tuple(mask.shape)} != {tuple(shape)}")
    return mask
