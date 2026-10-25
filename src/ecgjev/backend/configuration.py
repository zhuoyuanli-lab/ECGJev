import math

from ecgjev.models.decoder.startlux_decision.model import CAUSAL


def validate_adapter_settings(rank, alpha, dropout):
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("positive LoRA rank/alpha and dropout in [0,1) required")


def decoder_options(options, max_prompt_tokens):
    if type(max_prompt_tokens) is not int or max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be a positive integer")
    options = dict(options or {})
    allowed = {
        "device_map", "graphs", "max_batch_tokens", "max_length", "prefill_chunk",
        "images", "max_pixels", "min_pixels",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unsupported decoder_opts: {sorted(unknown)}")
    options["graphs"] = False
    options["images"] = True
    options["max_length"] = min(int(options.get("max_length", max_prompt_tokens)), max_prompt_tokens)
    if options["max_length"] < 1:
        raise ValueError("positive decoder max_length required")
    return options


def configure_language_body(body, gradient_checkpointing=False):
    body.config._attn_implementation = CAUSAL
    body.config.use_cache = False
    if gradient_checkpointing:
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
