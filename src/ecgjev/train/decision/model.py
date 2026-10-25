from torch import nn

from ecgjev.inference import predict_scores

from ..losses import decision_loss
from ..optim import decision_parameter_groups
from .batching import question_evidence


class DecisionModel(nn.Module):
    def __init__(self, fusion, backend):
        super().__init__()
        if fusion.decoder_dim != backend.decoder_dim:
            raise ValueError("fusion and decoder dimensions differ")
        self.fusion = fusion
        self.backend = backend

    def evidence_tokens(self, signal, visual, rows, *, record_indices=None,
                        signal_padding_mask=None, visual_padding_mask=None):
        signal, visual, masks = question_evidence(
            rows, signal, visual, record_indices, signal_padding_mask, visual_padding_mask,
        )
        question = self.backend.instruction_embeddings(rows).detach()
        return self.fusion(signal, visual, question, condition=True, **masks)["tokens"]

    def forward(self, signal, visual, rows, *, record_indices=None, batch_rows=False,
                signal_padding_mask=None, visual_padding_mask=None):
        tokens = self.evidence_tokens(
            signal, visual, rows, record_indices=record_indices,
            signal_padding_mask=signal_padding_mask, visual_padding_mask=visual_padding_mask,
        )
        return self.backend(rows, tokens, batch_rows=batch_rows).float()

    def loss(self, signal, visual, rows, targets, **kwargs):
        return decision_loss(self(signal, visual, rows, **kwargs), targets)

    def parameter_groups(self, *, lr_fusion=1e-4, lr_lora=5e-5):
        return decision_parameter_groups(
            self.fusion, self.backend, lr_fusion=lr_fusion, lr_lora=lr_lora,
        )

    def predict(self, signal, visual, rows, **kwargs):
        return predict_scores(self, signal, visual, rows, **kwargs)
