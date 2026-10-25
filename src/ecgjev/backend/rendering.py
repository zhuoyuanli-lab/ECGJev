import copy
from typing import Any, cast

import torch

from ecgjev.models.decoder.startlux_decision import jevfmt as J


def validate_binary_row(row):
    order = J.validate(row)
    if row["type"] != "noul" or len(order) != 2 or set(order) != {"true", "false"}:
        raise ValueError("backend requires binary true/false noul rows")
    if row.get("semantic_option_order", order) != order:
        raise ValueError("semantic_option_order must match the actual options order")
    if row["state"].count("<image>") != 1:
        raise ValueError("each row must contain exactly one <image> in its state")
    return order


def render_question(decision, row, order, dummy, max_prompt_tokens):
    decision._media = decision._encode_images([dummy])
    state = decision._media_state(row["state"], 1)
    rendered = cast(Any, decision._render(dict(row, state=state), order))
    if len(rendered) > max_prompt_tokens:
        raise ValueError(f"prompt length {len(rendered)} > {max_prompt_tokens}; no truncation")
    image = rendered.feature >= 0
    if not torch.equal(rendered.feature[image], torch.arange(dummy.shape[0])):
        raise ValueError("rendered image must map exactly to the row's K tokens")
    return rendered, image


def render_rows(decision, rows, k, decoder_dim, max_prompt_tokens):
    rendered, orders = [], []
    dummy = torch.zeros((k, decoder_dim), device=decision.device, dtype=torch.float32)
    for index, row in enumerate(rows):
        order = validate_binary_row(row)
        local, image = render_question(decision, row, order, dummy, max_prompt_tokens)
        ids = copy.copy(local)
        ids.feature = torch.where(image, local.feature + index * k, local.feature)
        rendered.append(ids)
        orders.append(order)
    return rendered, orders
