import torch


def validate_class_vectors(vectors, decoder_dim):
    if (vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] != decoder_dim
            or not torch.isfinite(vectors).all() or (vectors.float().norm(dim=-1) == 0).any()):
        raise ValueError("finite, nonzero descriptor vectors [C,decoder_dim] required")


def description_rows(descriptions):
    sizes, rows = [], []
    for description in descriptions:
        variants = [description] if isinstance(description, str) else list(description)
        if not variants or any(not isinstance(text, str) or not text.strip() for text in variants):
            raise ValueError("each class needs at least one nonempty description")
        sizes.append(len(variants))
        rows.extend({"instructions": text} for text in variants)
    if not rows:
        raise ValueError("at least one class is required")
    return rows, sizes


@torch.no_grad()
def descriptor_vectors(backend, descriptions):
    rows, sizes = description_rows(descriptions)
    embeddings = backend.instruction_embeddings(rows).detach()
    return torch.stack([part.mean(0) for part in embeddings.split(sizes)])


def initial_class_bias(vectors, prevalence):
    count = vectors.shape[0]
    bias = vectors.new_zeros(count)
    if prevalence is None:
        return bias
    prevalence = torch.as_tensor(prevalence, device=bias.device, dtype=bias.dtype)
    if (prevalence.shape != (count,) or not torch.isfinite(prevalence).all()
            or ((prevalence < 0) | (prevalence > 1)).any()):
        raise ValueError("prevalence must contain one fraction in [0,1] per class")
    return torch.logit(prevalence.clamp(1e-4, 1 - 1e-4))
