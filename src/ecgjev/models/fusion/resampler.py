from __future__ import annotations

import math

import torch
from torch import nn

from .attention import FusionBlock
from .positions import position_channels, time_positions
from .validation import validate_configuration, validate_question, validate_views


class DualViewResampler(nn.Module):
    def __init__(self, signal_dim=1024, visual_dim=768, decoder_dim=1024, width=256,
                 k=16, layers=2, heads=8, embedding_rms: float = .02, dropout: float = 0):
        super().__init__()
        dimensions = (signal_dim, visual_dim, decoder_dim, width, k, layers, heads)
        validate_configuration(dimensions, width, heads, embedding_rms, dropout)
        self.signal_dim, self.visual_dim, self.decoder_dim = signal_dim, visual_dim, decoder_dim
        self.width, self.k = width, k
        self.signal_proj = nn.Linear(signal_dim, width)
        self.visual_proj = nn.Linear(visual_dim, width)
        self.question_proj = nn.Linear(decoder_dim, width)
        self.lead = nn.Embedding(12, width)
        self.view = nn.Embedding(2, width)
        self.queries = nn.Parameter(torch.randn(k, width) * .02)
        self.q_scale = nn.Parameter(torch.tensor(.1))
        self.blocks = nn.ModuleList([FusionBlock(width, heads, dropout) for _ in range(layers)])
        self.out = nn.Linear(width, decoder_dim)
        self.out_norm = nn.LayerNorm(decoder_dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(embedding_rms)))

    def _encode_signal(self, features, mask):
        mask = mask.to(self.queries.device)
        clean = features.to(device=self.queries.device, dtype=self.queries.dtype)
        hidden = self.signal_proj(clean.masked_fill(mask[..., None], 0))
        position = position_channels(time_positions(mask), self.width).to(hidden.dtype)
        return hidden + position + self.view.weight[0], mask

    def _encode_spectrum(self, features, mask):
        mask = mask.to(self.queries.device)
        clean = features.to(device=self.queries.device, dtype=self.queries.dtype)
        hidden = self.visual_proj(clean.masked_fill(mask[..., None], 0))

        time = position_channels(time_positions(mask.all(dim=2)), self.width).to(hidden.dtype)
        coordinates = torch.linspace(100, 0, hidden.shape[2], device=self.queries.device)
        frequency = position_channels(coordinates, self.width, frequency=True).to(hidden.dtype)
        hidden = hidden + time[:, :, None] + frequency[None, None, :, None]
        hidden = hidden + self.lead.weight[None, :, None, None] + self.view.weight[1]
        return hidden.reshape(hidden.shape[0], -1, self.width), mask.reshape(mask.shape[0], -1)

    def _collect_evidence(self, signal, visual, signal_mask, visual_mask):
        parts, masks = [], []
        if signal is not None:
            hidden, mask = self._encode_signal(signal, signal_mask)
            parts.append(hidden)
            masks.append(mask)
        if visual is not None:
            hidden, mask = self._encode_spectrum(visual, visual_mask)
            parts.append(hidden)
            masks.append(mask)
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)

    def _condition_queries(self, batch, question, condition):
        queries = self.queries[None].expand(batch, -1, -1)
        if condition:
            question = question.to(device=self.queries.device, dtype=self.queries.dtype)
            queries = queries + self.q_scale * self.question_proj(question)[:, None]
        return queries

    def _resample(self, queries, features, padding_mask):
        weights = None
        for block in self.blocks:
            queries, weights = block(queries, features, padding_mask)
        return queries, weights

    def forward(self, signal_features, spectral_features, question, condition=True,
                signal_padding_mask=None, visual_padding_mask=None) -> dict:
        batch, signal_mask, visual_mask = validate_views(
            signal_features, spectral_features, self.signal_dim, self.visual_dim,
            signal_padding_mask, visual_padding_mask,
        )
        validate_question(question, batch, self.decoder_dim, condition)
        features, padding_mask = self._collect_evidence(
            signal_features, spectral_features, signal_mask, visual_mask,
        )
        queries = self._condition_queries(batch, question, condition)
        queries, weights = self._resample(queries, features, padding_mask)
        tokens = self.out_norm(self.out(queries)) * self.log_scale.exp()
        return {
            "tokens": tokens,
            "attention_auxiliary_only": weights,
            "signal_tokens": 0 if signal_features is None else signal_features.shape[1],
        }
