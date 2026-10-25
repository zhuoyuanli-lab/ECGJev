import torch

from ecgjev.models.decoder.startlux_decision.model import padded_length

from .readout import option_letter_logits


def validate_evidence(rows, tokens, decoder_dim):
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or tokens.shape[0] != len(rows) or tokens.shape[1] < 1
            or tokens.shape[2] != decoder_dim or not tokens.is_floating_point()):
        raise ValueError(f"tokens must be floating [len(rows),K,{decoder_dim}]")
    if not torch.isfinite(tokens).all():
        raise ValueError("evidence tokens must be finite")


def padded_token_ids(decision, rendered):
    length = min(padded_length(max(len(row) for row in rendered)), decision.max_length)
    ids = torch.full(
        (len(rendered), length), decision.pad, dtype=torch.long, device=decision.device,
    )
    for index, row in enumerate(rendered):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=decision.device)
    return ids


def run_rows(decision, rendered, evidence):
    ids = padded_token_ids(decision, rendered)
    decision._media = {"features": evidence}
    with decision._causal():
        result = decision._forward(ids, rendered, use_cache=False, return_dict=True)
    last = torch.tensor(
        [len(row) - 1 for row in rendered], device=decision.device, dtype=torch.long,
    )
    return option_letter_logits(result.last_hidden_state, last, decision.letter_rows)
