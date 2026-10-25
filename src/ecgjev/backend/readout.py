import torch


def option_letter_logits(hidden, last_positions, letter_rows):
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    final = hidden[batch, last_positions.to(hidden.device)].float()
    return final @ letter_rows[:2].detach().to(final.device).T


def semantic_logits(raw, orders):
    indices = torch.tensor(
        [[order.index("true"), order.index("false")] for order in orders],
        device=raw.device, dtype=torch.long,
    )
    return raw.gather(1, indices).float()
