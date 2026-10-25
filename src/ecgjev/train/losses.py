import torch
from torch.nn import functional as F


def masked_alignment_loss(logits, labels, label_mask=None):
    if labels.shape != logits.shape:
        raise ValueError("labels must match logits [B,C]")
    labels = labels.to(device=logits.device)
    if label_mask is None:
        label_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_mask.dtype != torch.bool or label_mask.shape != labels.shape:
        raise ValueError("label_mask must be boolean [B,C]")
    label_mask = label_mask.to(logits.device)
    values, targets = logits[label_mask].float(), labels[label_mask].float()
    if not targets.numel():
        raise ValueError("batch has no known labels")
    if not torch.isfinite(values).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("known labels must be binary and their logits finite")
    return F.binary_cross_entropy_with_logits(values, targets)


def decision_loss(logits, targets):
    if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] < 1:
        raise ValueError("nonempty logits [N,2] required")
    targets = torch.as_tensor(targets, device=logits.device)
    if targets.shape != (logits.shape[0],) or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("one binary target per question required")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return F.cross_entropy(logits.float(), 1 - targets.long())
