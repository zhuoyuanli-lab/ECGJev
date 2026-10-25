import re

from torch import nn


_LANGUAGE_LINEAR = re.compile(
    r"(?:.*\.)?layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def language_targets(body):
    targets = [
        name for name, module in body.named_modules()
        if isinstance(module, nn.Linear) and _LANGUAGE_LINEAR.fullmatch(name)
    ]
    if not targets:
        raise ValueError("no verified language linear modules found for LoRA")
    return targets


def set_adapter_trainable(body, trainable):
    for name, parameter in body.named_parameters():
        parameter.requires_grad_(bool(trainable and (".lora_A." in name or ".lora_B." in name)))
    body.get_input_embeddings().weight.requires_grad_(False)
