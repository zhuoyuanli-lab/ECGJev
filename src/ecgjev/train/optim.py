def decision_parameter_groups(fusion, backend, *, lr_fusion=1e-4, lr_lora=5e-5):
    lora = backend.trainable_parameters()
    if not lora:
        raise ValueError("S2 training requires a backend with trainable LoRA")
    return [
        {"params": [p for p in fusion.parameters() if p.requires_grad], "lr": lr_fusion},
        {"params": lora, "lr": lr_lora},
    ]
