from pathlib import Path
from typing import cast

import torch

from ecgjev.models.decoder.startlux_decision.model import CAUSAL

from .state import validate_adapter_config, validate_adapter_state
from .targets import language_targets, set_adapter_trainable


class LanguageAdapterMixin:
    def _mount_lora(self, settings):
        from peft import LoraConfig, get_peft_model
        targets = language_targets(self.decision.body)
        body = get_peft_model(self.decision.body, LoraConfig(target_modules=targets, **settings))
        if set(cast(list[str], body.base_model.targeted_module_names)) != set(targets):
            raise RuntimeError("PEFT mounted modules outside the explicit language target list")


        cast(LoraConfig, body.peft_config["default"]).target_modules = list(targets)
        self.lora_targets = targets
        self._set_body(body)
        self._set_adapter_trainable(True)

    def _set_body(self, body):
        self.decoder = body
        self.decision.body = body
        body.config.use_cache = False
        body.config._attn_implementation = CAUSAL

    def _set_adapter_trainable(self, trainable):
        set_adapter_trainable(self.decoder, trainable)

    def trainable_parameters(self) -> list:
        return [parameter for parameter in self.decoder.parameters() if parameter.requires_grad]

    def state_lora(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict
        if not hasattr(self.decoder, "peft_config"):
            return {}
        return {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(
            self.decoder, adapter_name="default", save_embedding_layers=False).items()}

    def load_state_lora(self, state):
        from peft import set_peft_model_state_dict
        current = self.state_lora()
        validate_adapter_state(state, current)
        if not current:
            return
        result = set_peft_model_state_dict(self.decoder, state, adapter_name="default")
        if result.unexpected_keys:
            raise ValueError(f"unexpected LoRA keys: {result.unexpected_keys}")
        restored = self.state_lora()
        if any(not torch.equal(restored[key], value.detach().cpu()) for key, value in state.items()):
            raise RuntimeError("LoRA state did not restore exactly")

    def save_lora(self, path):
        if not hasattr(self.decoder, "peft_config"):
            raise ValueError("no LoRA adapter to save")
        self.decoder.save_pretrained(Path(path), safe_serialization=True, save_embedding_layers=False)

    def load_lora(self, path, trainable=False):
        from peft import LoraConfig
        from peft.utils.save_and_load import load_peft_weights
        path = Path(path).expanduser().resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing local LoRA config: {path}")
        config = cast(LoraConfig, LoraConfig.from_pretrained(str(path), local_files_only=True))

        base = self.decoder.get_base_model() if hasattr(self.decoder, "peft_config") else self.decoder
        if self.lora_targets:
            allowed = set(self.lora_targets)
        else:
            allowed = set(language_targets(base))
        settings = validate_adapter_config(config, allowed)
        if hasattr(self.decoder, "peft_config"):
            existing = self.decoder.peft_config["default"]
            if any(getattr(existing, key) != value for key, value in settings.items()):
                raise ValueError("LoRA rank/alpha/dropout differ from the mounted adapter")
        else:
            self._mount_lora(settings)
            self._configure_checkpointing()
        state = load_peft_weights(str(path), device="cpu", local_files_only=True)
        self.load_state_lora(state)
        self._set_adapter_trainable(trainable)
        self.train(bool(trainable))
        self._disable_reuse()
        return self
