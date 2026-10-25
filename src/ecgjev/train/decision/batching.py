import torch

from ecgjev.models.evidence import detached_view


def question_record_indices(rows, signal, visual, record_indices):
    reference = signal if signal is not None else visual
    if reference is None:
        raise ValueError("at least one evidence view is required")
    if not rows:
        raise ValueError("at least one question is required")
    count = reference.shape[0]
    if record_indices is None:
        if len(rows) != count:
            raise ValueError("record_indices required for multiple questions per record")
        record_indices = torch.arange(count)
    indices = torch.as_tensor(record_indices)
    if (indices.ndim != 1 or indices.numel() != len(rows)
            or indices.dtype not in (torch.int32, torch.int64)
            or ((indices < 0) | (indices >= count)).any()):
        raise ValueError("one valid integer record index per question required")
    return indices, count


def select_evidence_tensor(value, indices, count):
    if value is None:
        return None
    if value.shape[0] != count:
        raise ValueError("feature and padding-mask batch sizes must agree")
    return detached_view(value).index_select(0, indices.to(value.device))


def question_evidence(rows, signal, visual, record_indices, signal_mask, visual_mask):
    indices, count = question_record_indices(rows, signal, visual, record_indices)
    signal = select_evidence_tensor(signal, indices, count)
    visual = select_evidence_tensor(visual, indices, count)
    masks = {
        "signal_padding_mask": select_evidence_tensor(signal_mask, indices, count),
        "visual_padding_mask": select_evidence_tensor(visual_mask, indices, count),
    }
    return signal, visual, masks
