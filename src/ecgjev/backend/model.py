from pathlib import Path
import threading

import torch
from torch import nn

from ecgjev.models.decoder.ecg_decision import ECGDecision
from ecgjev.models.decoder.startlux_decision.model import _SHARED

from .adapters.language import LanguageAdapterMixin
from .configuration import configure_language_body, decoder_options, validate_adapter_settings
from .embeddings import embedding_rms, instruction_embeddings
from .forward import run_rows, validate_evidence
from .readout import semantic_logits
from .rendering import render_rows


class DualViewBackend(LanguageAdapterMixin, nn.Module):
    def __init__(self, decoder_path, device="cpu", decoder_opts=None, lora=True,
                 rank=16, alpha=32, dropout=.05, gradient_checkpointing=False,
                 max_prompt_tokens=768):
        super().__init__()
        options = decoder_options(decoder_opts, max_prompt_tokens)
        validate_adapter_settings(rank, alpha, dropout)
        self.decoder_path = Path(decoder_path).expanduser().resolve()
        self.max_prompt_tokens = max_prompt_tokens
        self.decision = ECGDecision(str(self.decoder_path), device=device, **options)
        self._freeze_base()
        self.lora_targets: list[str] = []
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._lora_settings = {
            "r": rank, "lora_alpha": alpha, "lora_dropout": dropout, "bias": "none",
        }
        self._lock = threading.RLock()
        self._disable_reuse()
        configure_language_body(self.decoder)
        if lora:
            self._mount_lora(self._lora_settings)
        self._configure_checkpointing()
        self.train()

    def _freeze_base(self):
        decision = self.decision
        if decision.vision is None:
            raise ValueError(f"soft evidence requires multimodal positions: {decision.vision_note}")
        decision.vision.requires_grad_(False)
        decision.body.requires_grad_(False)
        self.decoder = decision.body
        self.decoder_dim = int(self.decoder.get_input_embeddings().embedding_dim)
        self.embedding_rms = embedding_rms(self.decoder.get_input_embeddings().weight)
        decision.letter_rows = decision.letter_rows.detach().clone().float()

    def _disable_reuse(self):
        decision = self.decision
        decision.share = False
        decision.graphs.clear()
        decision._media = decision._kept = None
        _SHARED.kv = None

    def _configure_checkpointing(self):
        configure_language_body(self.decoder, self.gradient_checkpointing)

    @property
    def tokenizer(self):
        return self.decision.tok

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError("training mode must be boolean")
        if hasattr(self, "decision") and self.decision.vision is not None:
            self.decision.vision.eval()
        return super().train(mode)

    def instruction_embeddings(self, rows):
        return instruction_embeddings(self.decoder, self.tokenizer, rows)

    def _render_rows(self, rows, k):
        return render_rows(self.decision, rows, k, self.decoder_dim, self.max_prompt_tokens)

    def _run_rows(self, rendered, evidence):
        return run_rows(self.decision, rendered, evidence)

    def _evidence_logits(self, rendered, evidence, batch_rows):
        if batch_rows:
            return self._run_rows(rendered, evidence)

        return torch.cat([self._run_rows([row], evidence) for row in rendered], dim=0)

    def forward(self, rows, tokens, batch_rows=False):
        with self._lock:
            self._disable_reuse()
            try:
                validate_evidence(rows, tokens, self.decoder_dim)
                if not rows:
                    return tokens.new_empty((0, 2), dtype=torch.float32)
                rendered, orders = self._render_rows(rows, tokens.shape[1])
                evidence = tokens.to(self.decision.device).reshape(-1, self.decoder_dim)
                raw = self._evidence_logits(rendered, evidence, batch_rows)
                return semantic_logits(raw, orders)
            finally:
                self._disable_reuse()
