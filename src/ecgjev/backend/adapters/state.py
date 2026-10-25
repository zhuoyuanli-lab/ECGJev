import torch


def validate_adapter_state(state, current):
    if not isinstance(state, dict) or set(state) != set(current):
        raise ValueError("LoRA state must contain exactly the current adapter keys")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != current[key].shape
                or value.dtype != current[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"LoRA state shape/dtype/values mismatch: {key}")


def validate_adapter_config(config, allowed_targets):
    targets = config.target_modules
    if (not isinstance(targets, (set, list)) or set(targets) != allowed_targets or config.bias != "none"
            or config.modules_to_save or config.use_dora or config.use_rslora
            or config.rank_pattern or config.alpha_pattern):
        raise ValueError("LoRA config must match the explicit language-only adapter architecture")
    return {
        "r": config.r, "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout, "bias": "none",
    }
