import math

import torch

from .masking import normalize_padding_mask


def validate_configuration(dimensions, width, heads, embedding_rms, dropout):
    if (any(type(value) is not int or value < 1 for value in dimensions)
            or width % 4 or width % heads):
        raise ValueError("positive dimensions required; width must divide by four and heads")
    if not math.isfinite(embedding_rms) or embedding_rms <= 0 or not 0 <= dropout < 1:
        raise ValueError("positive finite embedding_rms and dropout in [0,1) required")


def validate_question(question, batch, decoder_dim, condition):
    if question is not None and (question.ndim != 2 or question.shape != (batch, decoder_dim)):
        raise ValueError(f"question must be [B,{decoder_dim}]")
    if condition and (question is None or not torch.isfinite(question).all()):
        raise ValueError("conditioning requires finite instructions embeddings")


def validate_views(signal, visual, signal_dim, visual_dim, signal_mask, visual_mask):
    if signal is None and visual is None:
        raise ValueError("at least one view is required")
    reference = signal if signal is not None else visual
    assert reference is not None
    batch = reference.shape[0]
    if batch < 1:
        raise ValueError("empty feature batch")
    if signal is not None and (signal.ndim != 3 or signal.shape[-1] != signal_dim or signal.shape[1] < 1):
        raise ValueError(f"signal must be [B,Ts,{signal_dim}] with nonempty Ts")
    if visual is not None and (visual.ndim != 5 or visual.shape[:2] != (batch, 12)
                               or visual.shape[-1] != visual_dim or min(visual.shape[2:4]) < 1):
        raise ValueError(f"spectrum must be [B,12,Fp,Tp,{visual_dim}]")
    if signal is not None and signal.shape[0] != batch:
        raise ValueError("view batch mismatch")
    if signal is not None and visual is not None and signal.device != visual.device:
        raise ValueError("views must be on the same device")
    sm, vm = normalize_padding_mask(signal, signal_mask), normalize_padding_mask(visual, visual_mask, visual=True)
    for features, mask in ((signal, sm), (visual, vm)):
        if features is not None:
            assert mask is not None
            if not features.is_floating_point():
                raise ValueError("features must be floating point")
            if not torch.isfinite(features.masked_fill(mask[..., None], 0)).all():
                raise ValueError("unmasked features must be finite")
    device = reference.device
    present = torch.zeros(batch, device=device, dtype=torch.bool)
    for mask in (sm, vm):
        if mask is not None:
            present |= (~mask).reshape(batch, -1).any(-1)
    if not present.all():
        raise ValueError("every row must have at least one unmasked evidence token")
    return batch, sm, vm
