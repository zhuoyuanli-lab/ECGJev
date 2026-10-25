import math

import torch
from torch import nn
from torch.nn import functional as F

from ecgjev.models.evidence import detached_view

from ..losses import masked_alignment_loss
from .descriptors import descriptor_vectors, initial_class_bias, validate_class_vectors


class DescriptorAlignment(nn.Module):
    def __init__(self, fusion, class_vectors, *, prevalence=None, temperature=.07):
        super().__init__()
        validate_class_vectors(class_vectors, fusion.decoder_dim)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.fusion = fusion
        self.temperature = float(temperature)
        self.register_buffer("class_vectors", F.normalize(class_vectors.detach().float(), dim=-1))
        self.bias = nn.Parameter(initial_class_bias(self.class_vectors, prevalence))

    @classmethod
    @torch.no_grad()
    def from_descriptions(cls, fusion, backend, descriptions, **kwargs):
        return cls(fusion, descriptor_vectors(backend, descriptions), **kwargs)

    def pooled_evidence(self, signal, visual, **padding_masks):
        tokens = self.fusion(
            detached_view(signal), detached_view(visual), None, condition=False, **padding_masks,
        )["tokens"]
        return F.normalize(tokens.float().mean(1), dim=-1)

    def forward(self, signal, visual, **padding_masks):
        pooled = self.pooled_evidence(signal, visual, **padding_masks)
        return pooled @ self.class_vectors.T / self.temperature + self.bias

    def loss(self, signal, visual, labels, label_mask=None, **padding_masks):
        logits = self(signal, visual, **padding_masks)
        return masked_alignment_loss(logits, labels, label_mask)
