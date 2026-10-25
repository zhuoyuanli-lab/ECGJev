import math

import torch


def sinusoidal(values, width):
    if width % 2:
        raise ValueError("sinusoidal width must be even")
    rates = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=torch.float32)
                      * (-math.log(10000) / width))
    phase = values.float().unsqueeze(-1) * rates
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)

def time_positions(mask):
    index = torch.arange(mask.shape[-1], device=mask.device)
    last = torch.where(~mask, index, 0).amax(-1, keepdim=True).clamp_min(1)
    return index.float() * (100. / last.float())


def position_channels(values, width, *, frequency=False):
    encoded = sinusoidal(values, width // 2)
    zero = torch.zeros_like(encoded)
    return torch.cat((zero, encoded) if frequency else (encoded, zero), dim=-1)
