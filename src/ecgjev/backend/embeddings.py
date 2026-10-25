import math

import torch


@torch.no_grad()
def embedding_rms(weight):
    square_sum = sum(
        chunk.float().square().sum(dtype=torch.float64).item()
        for chunk in weight.detach().split(4096)
    )
    value = float(math.sqrt(square_sum / weight.numel()))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("decoder embeddings must have positive finite RMS")
    return value


@torch.no_grad()
def instruction_embeddings(body, tokenizer, rows):
    embedding = body.get_input_embeddings()
    vectors = []
    for row in rows:
        instructions = row.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("nonempty instructions required")
        ids = tokenizer.encode(instructions, add_special_tokens=False)
        if not ids:
            raise ValueError("instructions tokenize to an empty sequence")
        values = embedding(torch.tensor(ids, device=embedding.weight.device, dtype=torch.long))
        vectors.append(values.detach().float().mean(dim=0))
    if not vectors:
        return torch.empty((0, embedding.embedding_dim), device=embedding.weight.device, dtype=torch.float32)
    return torch.stack(vectors).detach()
