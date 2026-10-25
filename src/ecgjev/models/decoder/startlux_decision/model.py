import base64
import binascii
import contextlib
import functools
import importlib
import io
import json
import os
import re
import threading
import inspect
import warnings

import torch
import torch.nn.functional as F

from . import jevfmt as J

__all__ = ["StartLuxDecision", "load_model", "load_image", "fast_kernels_active", "choice_confidence", "score_confidence"]

GRAPH_LENGTHS = (128, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)
GRAPH_ROWS = (1, 2, 3, 4)
MULTI_ROW_MAX_LENGTH = 1024


PAD_LENGTHS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
               5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536)


SHARE_MIN = 4096
SHARE_STEP = 1024
PREFILL_CHUNK = 32768
CAUSAL = "startlux_causal"
_SHARED = threading.local()
_LSE_KERNEL = []
IMAGE = "<image>"
_SLOT = "\x00IMAGE\x00"


def load_image(x, paths=True):
    from PIL import Image, UnidentifiedImageError

    if isinstance(x, Image.Image):
        return x.convert("RGB")
    raw = x
    if isinstance(x, str):
        if paths and not x.startswith("data:") and os.path.isfile(x):
            with open(x, "rb") as f:
                raw = f.read()
        else:
            try:
                raw = base64.b64decode(re.sub(r"\s+", "", x.split(",", 1)[1] if x.startswith("data:") else x),
                                      validate=True)
            except (binascii.Error, IndexError):
                raise ValueError("an image must be base64 or a data URI" + (", a file path" if paths else "") +
                                 " or bytes") from None
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"unsupported image value of type {type(x).__name__}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"not a readable image: {e}") from None


class _Row(list):
    pass


def choice_confidence(p):
    n = len(p)
    return 1.0 if n < 2 else min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def score_confidence(p):
    n = len(p)
    m = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - m) for i, v in enumerate(p))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even) if even else 1.0


def padded_length(n):
    for length in PAD_LENGTHS:
        if n <= length:
            return length
    return -(-n // 8192) * 8192


def _attend(query, key, value, causal, scale):
    if not _LSE_KERNEL:
        for kind in ("cudnn", "flash", None):
            try:
                out = _attend_with(kind, query, key, value, causal, scale)
            except RuntimeError:
                continue
            _LSE_KERNEL.append(kind)
            return out
    return _attend_with(_LSE_KERNEL[0], query, key, value, causal, scale)


def _attend_with(kind, query, key, value, causal, scale):
    if kind is None:
        return None
    if kind == "cudnn":
        out = torch.ops.aten._scaled_dot_product_cudnn_attention(query, key, value, None, True, 0.0, causal, False,
                                                                  scale=scale)
    else:
        groups = query.shape[1] // key.shape[1]
        if groups > 1:
            key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
        out = torch.ops.aten._scaled_dot_product_flash_attention(query, key, value, 0.0, causal, False, scale=scale)
    return out[0], out[1].reshape(query.shape[:3])


def _merge(a, b):
    (out_a, lse_a), (out_b, lse_b) = a, b
    top = torch.maximum(lse_a, lse_b)
    wa, wb = (lse_a - top).exp().unsqueeze(-1), (lse_b - top).exp().unsqueeze(-1)
    return ((out_a.float() * wa + out_b.float() * wb) / (wa + wb)).to(out_a.dtype)


def _causal_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from torch.nn.attention.bias import causal_lower_right

    rows, heads, q, dim = query.shape
    shared = getattr(_SHARED, "kv", None)
    shared = shared.get(module.layer_idx) if shared else None
    if shared is not None:
        folded = query.transpose(0, 1).reshape(1, heads, rows * q, dim)
        part = _attend(folded, shared[0], shared[1], False, scaling)
        if part is not None:
            out_a = part[0].reshape(heads, rows, q, dim).transpose(0, 1)
            lse_a = part[1].reshape(heads, rows, q).transpose(0, 1)
            out = _merge((out_a, lse_a), _attend(query, key, value, True, scaling))
            return out.transpose(1, 2).contiguous(), None
        key = torch.cat([shared[0].expand(rows, -1, -1, -1), key], dim=-2)
        value = torch.cat([shared[1].expand(rows, -1, -1, -1), value], dim=-2)
    k = key.shape[2]
    if q == k:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=q > 1, scale=scaling, enable_gqa=True)
    else:
        before = _attend(query, key[:, :, :k - q], value[:, :, :k - q], False, scaling)
        if before is not None:
            out = _merge(before, _attend(query, key[:, :, k - q:], value[:, :, k - q:], True, scaling))
        else:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=causal_lower_right(q, k), scale=scaling,
                                                 enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def _register_causal_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(CAUSAL, _causal_attention)
    AttentionMaskInterface.register(CAUSAL, lambda *args, **kwargs: None)


def _common_length(a, b):
    n = min(len(a), len(b))
    diff = (torch.tensor(a[:n]) != torch.tensor(b[:n])).nonzero()
    return int(diff[0, 0]) if len(diff) else n


@functools.cache
def _shared_kv_layer():
    from transformers.cache_utils import DynamicLayer

    class SharedKV(DynamicLayer):
        def __init__(self, length):
            super().__init__()
            self.length, self.end, self.frozen = length, 0, False

        def update(self, key_states, value_states, *args, **kwargs):
            if self.frozen:
                return key_states, value_states
            if not self.is_initialized:
                b, h, _, d = key_states.shape
                self.keys = key_states.new_empty((b, h, self.length, d))
                self.values = value_states.new_empty((b, h, self.length, d))
                self.dtype, self.device, self.is_initialized = key_states.dtype, key_states.device, True
            n = key_states.shape[-2]
            self.keys[:, :, self.end:self.end + n] = key_states
            self.values[:, :, self.end:self.end + n] = value_states
            self.end += n
            return self.keys[:, :, :self.end], self.values[:, :, :self.end]

        def get_seq_length(self):
            return self.end

        def get_mask_sizes(self, query_length):
            return self.end + query_length, 0

    return SharedKV


def fast_kernels_active(path):
    from transformers import AutoConfig

    kind = AutoConfig.from_pretrained(path).model_type
    try:
        modeling = importlib.import_module(f"transformers.models.{kind}.modeling_{kind}")
    except ImportError:
        return False
    if getattr(modeling, "is_fast_path_available", False):
        return True


    try:
        importlib.import_module("fla")
        importlib.import_module("causal_conv1d")
    except ImportError:
        return False
    src = inspect.getsource(modeling)
    return "use_kernel_func_from_hub_with_fallback" in src


def load_model(path, device, device_map=None):
    import transformers

    config = transformers.AutoConfig.from_pretrained(path)
    extra = {"experts_implementation": "grouped_mm"} if getattr(config.get_text_config(), "num_experts", 0) else {}
    placement = "auto" if str(device_map or "").lower() == "auto" else {"": device}
    model = getattr(transformers, config.architectures[0]).from_pretrained(path, dtype=torch.bfloat16,
                                                                           device_map=placement, **extra)
    return model, getattr(model.model, "language_model", model.model)


class StartLuxDecision:
    def __init__(self, path, device=None, max_length=262144, max_batch_tokens=65536, graphs=True,
                 prefill_chunk=PREFILL_CHUNK, images=True, max_pixels=1 << 20, min_pixels=65536,
                 device_map=None):
        from transformers import AutoTokenizer

        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path}: expected a local StartLux-Decision directory (weights are shared separately)")
        cfg = json.load(open(os.path.join(path, "decision_config.json")))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.letters = J.check_tokenizer(self.tok)
        if self.letters != cfg["letter_token_ids"]:
            raise ValueError("tokenizer letter ids differ from decision_config.json")
        self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        wide = cfg.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fast_kernels = fast_kernels_active(path)
        if self.device.type == "cuda" and not self.fast_kernels:
            msg = ("flash-linear-attention and causal-conv1d are not active, so transformers would run the plain torch "
                   "path of the linear-attention layers (>10x slower). pip install flash-linear-attention causal-conv1d, "
                   "or set STARTLUX_ALLOW_SLOW=1 to run anyway.")
            if os.environ.get("STARTLUX_ALLOW_SLOW") != "1":
                raise RuntimeError(msg)
            warnings.warn(msg)


        self.sharded = str(device_map or "").lower() == "auto"
        model, body = load_model(path, self.device, device_map=device_map)
        self.body = body.eval()
        if self.sharded:
            self.device = self.body.get_input_embeddings().weight.device
        head = model.get_output_embeddings().weight


        self.letter_rows = head.index_select(0, torch.tensor(self.letters, device=head.device)).float().detach()
        self.vision, self.image_processor, self.vision_note = None, None, "the checkpoint has no vision tower"
        if not images:
            self.vision_note = "the model was loaded with images=False"
        elif getattr(model.model, "visual", None) is not None:
            try:
                from transformers import AutoImageProcessor

                self.image_processor = AutoImageProcessor.from_pretrained(path)
                self.image_processor.size = {"longest_edge": int(max_pixels), "shortest_edge": int(min_pixels)}
                self.vision = model.model.eval()
                self.image_token = model.config.image_token_id
            except (OSError, ImportError, ValueError) as e:
                self.vision_note = f"no image processor ({str(e).splitlines()[0][:200]})"
        del model
        self._media = None
        self.pad = self.tok.pad_token_id
        self.max_length, self.max_batch_tokens = int(max_length), int(max_batch_tokens)
        self.share, self.prefill_chunk = True, int(prefill_chunk)


        self.share_min, self.share_step = SHARE_MIN, SHARE_STEP
        self._kept = None
        _register_causal_attention()
        self.graphs = {}


        if self.sharded and graphs:
            warnings.warn("device_map='auto' shards the decoder over several GPUs; CUDA-graph "
                          "capture needs a single device, so the graphs stay off (eager path)")
            graphs = False
        if graphs and self.device.type == "cuda" and os.environ.get("STARTLUX_GRAPHS", "1") != "0":
            self._capture()


    def _slot_logits(self, ids, last):
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            hidden = self.body(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        h = hidden[torch.arange(ids.shape[0], device=ids.device), last].float()
        return h @ self.letter_rows.T

    @torch.inference_mode()
    def _capture(self):
        shapes = [(b, n) for b in GRAPH_ROWS for n in GRAPH_LENGTHS if b == 1 or n <= MULTI_ROW_MAX_LENGTH]
        pool = None
        for b, n in sorted(shapes, key=lambda s: (-s[0] * s[1], -s[1])):
            ids = torch.full((b, n), self.pad, dtype=torch.long, device=self.device)
            last = torch.full((b,), n - 1, dtype=torch.long, device=self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._slot_logits(ids, last)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self._slot_logits(ids, last)
            pool = graph.pool()
            self.graphs[(b, n)] = (graph, ids, last, out)

    def _graph_length(self, enc):
        longest = max(len(t) for _, t, _ in enc)
        fits = [n for b, n in self.graphs if b == len(enc) and n >= longest]
        return min(fits) if fits else None

    @torch.inference_mode()
    def _graph_logits(self, enc, n):
        graph, static_ids, last, out = self.graphs[(len(enc), n)]
        ids = torch.full((len(enc), n), self.pad, dtype=torch.long)
        for j, (_, t, _) in enumerate(enc):
            ids[j, :len(t)] = torch.tensor(t)
        static_ids.copy_(ids)
        last.copy_(torch.tensor([len(t) - 1 for _, t, _ in enc]))
        graph.replay()
        res = out.cpu()
        return [res[j, :c].clone() for j, (_, _, c) in enumerate(enc)]


    def _media_state(self, state, n):
        text = J.state_text(state)
        marks = text.count(IMAGE)
        if marks == n:
            return text.replace(IMAGE, _SLOT)
        if marks:
            raise ValueError(f"the state has {marks} {IMAGE} marks for {n} images")
        head = _SLOT if n == 1 else "\n".join(f"Image {k + 1}: {_SLOT}" for k in range(n))
        return head + ("\n\n" + text if text.strip() else "")

    @torch.no_grad()
    def _encode_images(self, images):
        if getattr(self, "vision", None) is None:
            raise ValueError(f"this model takes no images: {getattr(self, 'vision_note', 'this backend reads text only')}")
        enc = self.image_processor(images=[load_image(x) for x in images], return_tensors="pt")
        grid = enc["image_grid_thw"]
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            out = self.vision.get_image_features(enc["pixel_values"].to(self.device), grid.to(self.device),
                                                 return_dict=True)
        merge = self.vision.visual.spatial_merge_size
        return {"features": torch.cat(out.pooler_output, 0), "grid": grid, "tokens": (grid.prod(-1) // merge ** 2).tolist()}

    def _render(self, row, order):
        if self._media is None:
            return J.render_ids(row, self.tok, order, max_length=self.max_length)[0]
        msgs, _ = J.messages(row, order)
        content, parts = [], msgs[1]["content"].split(_SLOT)
        for k, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if k < len(parts) - 1:
                content.append({"type": "image"})
        chat = [{"role": "system", "content": [{"type": "text", "text": msgs[0]["content"]}]},
                {"role": "user", "content": content}]
        text = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not text.endswith(J.THINK_OFF_SUFFIX):
            raise ValueError("chat template lacks the thinking-off assistant prefix")
        counts, ids, k = self._media["tokens"], _Row(), 0
        for t in self.tok.encode(text, add_special_tokens=False):
            if t == self.image_token:
                if k == len(counts):
                    raise ValueError("more image places than images")
                ids.extend([t] * counts[k])
                k += 1
            else:
                ids.append(t)
        if k != len(counts):
            raise ValueError("fewer image places than images")
        if len(ids) > self.max_length:
            raise ValueError(f"length {len(ids)} > {self.max_length}")
        t = torch.tensor(ids)
        image = t == self.image_token
        pos, _ = self.vision.get_rope_index(t[None], mm_token_type_ids=image.int()[None],
                                            image_grid_thw=self._media["grid"])
        ids.pos = pos[:, 0]
        ids.feature = torch.where(image, image.long().cumsum(0) - 1, -1)
        return ids

    def _forward(self, ids, rows, offset=0, **kwargs):
        if self._media is None:
            return self.body(input_ids=ids.to(self.device), **kwargs)
        ids = ids.to(self.device)
        embeds = self.body.get_input_embeddings()(ids)


        if embeds.requires_grad and embeds.grad_fn is None:
            embeds = embeds.clone()
        features, n = self._media["features"], ids.shape[1]
        pos = torch.empty((3, ids.shape[0], n), dtype=torch.long)
        for j, t in enumerate(rows):
            p = t.pos[:, offset:offset + n]
            pos[:, j, :p.shape[1]] = p
            pos[:, j, p.shape[1]:] = p[:, -1:] + torch.arange(1, n - p.shape[1] + 1)
            f = t.feature[offset:offset + n]
            at = (f >= 0).nonzero().squeeze(1)
            if len(at):
                embeds[j, at.to(self.device)] = features[f[at].to(self.device)].to(embeds.dtype)
        return self.body(inputs_embeds=embeds, position_ids=pos.to(self.device), **kwargs)


    @torch.no_grad()
    def _logits(self, rows):
        enc = []
        for i, r in enumerate(rows):
            order = [o["id"] for o in r["options"]]
            enc.append((i, self._render(r, order), len(order)))
        out, tokens = [None] * len(rows), 0
        n = self._graph_length(enc) if self.graphs and enc and self._media is None else None
        if n is not None:
            for (i, t, _), z in zip(enc, self._graph_logits(enc, n)):
                out[i] = z
                tokens += len(t)
            return out, tokens
        groups, enc = self._shared_groups(enc) if self.share else ([], enc)
        for c, group in groups:
            for (i, t, _), z in zip(group, self._shared_logits(group, c)):
                out[i] = z
                tokens += len(t)
        enc.sort(key=lambda x: -len(x[1]))
        start = 0
        while start < len(enc):
            longest = padded_length(len(enc[start][1]))
            n = max(1, min(len(enc) - start, self.max_batch_tokens // longest))
            batch = enc[start:start + n]
            start += n
            ids = torch.full((len(batch), longest), self.pad, dtype=torch.long)
            for j, (_, t, _) in enumerate(batch):
                ids[j, :len(t)] = torch.tensor(t)
                tokens += len(t)
            with torch.autocast(self.device.type, dtype=torch.bfloat16), self._causal():
                hidden = self._forward(ids, [t for _, t, _ in batch], use_cache=False, return_dict=True).last_hidden_state
            pos = torch.tensor([len(t) - 1 for _, t, _ in batch], device=self.device)
            h = hidden[torch.arange(len(batch), device=self.device), pos].float()
            logits = h @ self.letter_rows.T
            for j, (i, _, c) in enumerate(batch):
                out[i] = logits[j, :c].cpu()
        return out, tokens

    @contextlib.contextmanager
    def _causal(self):
        config = self.body.config
        before, config._attn_implementation = config._attn_implementation, CAUSAL
        try:
            yield
        finally:
            config._attn_implementation = before


    def _shared_groups(self, enc):
        by_start, rest = {}, []
        for e in enc:
            if len(e[1]) > self.share_min:
                by_start.setdefault(tuple(e[1][:self.share_min]), []).append(e)
            else:
                rest.append(e)
        groups = []
        for group in by_start.values():
            first = group[0][1]
            c = min(len(t) for _, t, _ in group) - 1
            for _, t, _ in group[1:]:
                c = min(c, _common_length(first, t))
            c = c // self.share_step * self.share_step
            kept = self._kept[0] if self._kept else ()
            if (kept and all(len(t) > len(kept) and tuple(t[:len(kept)]) == kept for _, t, _ in group)
                    and max(len(t) for _, t, _ in group) - len(kept) <= self.prefill_chunk):
                c = len(kept)
            if c >= self.share_min and (len(group) > 1 or len(first) > self.prefill_chunk):
                groups.append((c, group))
            else:
                rest.extend(group)
        return groups, rest

    def _rows_cache(self, shared, rows):
        from transformers import DynamicCache
        from transformers.cache_utils import LinearAttentionLayer

        cache = DynamicCache(config=self.body.config)
        for i, layer in enumerate(shared.layers):
            if not isinstance(layer, LinearAttentionLayer):
                cache.layers[i] = layer
                continue
            copy = LinearAttentionLayer()
            for name in ("conv_states", "recurrent_states"):
                x = getattr(layer, name)
                setattr(copy, name, x.repeat(rows, *[1] * (x.dim() - 1)))
            copy.dtype, copy.device = layer.dtype, layer.device
            copy.max_batch_size, copy.conv_kernel_size = rows, layer.conv_kernel_size
            copy.is_conv_states_initialized = copy.is_recurrent_states_initialized = copy.has_previous_state = True
            cache.layers[i] = copy
        return cache

    @torch.no_grad()
    def _shared_logits(self, group, c):
        from transformers import DynamicCache
        from transformers.cache_utils import LinearAttentionLayer

        SharedKV = _shared_kv_layer()
        prefix = tuple(group[0][1][:c])
        out = []
        try:
            with torch.autocast(self.device.type, dtype=torch.bfloat16), self._causal():
                if self._kept and self._kept[0] == prefix:
                    shared = self._kept[1]
                else:
                    self._kept = None
                    shared = DynamicCache(config=self.body.config)
                    for i, layer in enumerate(shared.layers):
                        if not isinstance(layer, LinearAttentionLayer):
                            shared.layers[i] = SharedKV(c)
                    ids = torch.tensor([prefix])
                    for start in range(0, c, self.prefill_chunk):
                        self._forward(ids[:, start:start + self.prefill_chunk], [group[0][1]], start,
                                      past_key_values=shared, use_cache=True)
                    del ids
                    self._kept = (prefix, shared)
                _SHARED.kv = {}
                for i, layer in enumerate(shared.layers):
                    if isinstance(layer, SharedKV):
                        layer.frozen = True
                        _SHARED.kv[i] = (layer.keys[:, :, :layer.end], layer.values[:, :, :layer.end])
                tails = [(t, k) for _, t, k in group]
                per_pass = max(1, self.max_batch_tokens // padded_length(max(len(t) - c for t, _ in tails)))
                for start in range(0, len(tails), per_pass):
                    batch = tails[start:start + per_pass]
                    n = padded_length(max(len(t) - c for t, _ in batch))
                    ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
                    for j, (t, _) in enumerate(batch):
                        ids[j, :len(t) - c] = torch.tensor(t[c:])
                    cache = self._rows_cache(shared, len(batch))
                    hidden = self._forward(ids, [t for t, _ in batch], c, past_key_values=cache,
                                           use_cache=True).last_hidden_state
                    last = torch.tensor([len(t) - c - 1 for t, _ in batch], device=self.device)
                    logits = hidden[torch.arange(len(batch), device=self.device), last].float() @ self.letter_rows.T
                    out.extend(logits[j, :k].cpu() for j, (_, k) in enumerate(batch))
        finally:
            _SHARED.kv = None
        return out

    def _probs(self, rows):
        logits, tokens = self._logits(rows)
        return [torch.softmax(z / self.temperature.get(r["type"], 1.0), -1).tolist() for r, z in zip(rows, logits)], tokens

    def self_test(self, state, questions):
        if not self.graphs:
            return 0.0
        rows = [J.from_systemone(state, q) for q in questions.values()]
        graphed, _ = self._probs(rows)
        graphs, self.graphs = self.graphs, {}
        try:
            eager, _ = self._probs(rows)
        finally:
            self.graphs, self._kept = graphs, None
        return max(abs(a - b) for pg, pe in zip(graphed, eager) for a, b in zip(pg, pe))

    def _wide(self, state, spec):
        keys = list(spec["criteria"])
        n_groups = -(-len(keys) // self.group)
        size, extra = divmod(len(keys), n_groups)
        groups, start = [], 0
        for g in range(n_groups):
            end = start + size + (1 if g < extra else 0)
            groups.append(keys[start:end])
            start = end
        rows = [J.from_systemone(state, dict(spec, criteria={k: spec["criteria"][k] for k in g})) for g in groups]
        first, tokens = self._probs(rows)
        first_p = {k: p for g, ps in zip(groups, first) for k, p in zip(g, ps)}
        finalists = [k for g, ps in zip(groups, first) for k, _ in sorted(zip(g, ps), key=lambda x: -x[1])[:self.keep]]
        fin_spec = dict(spec, criteria={k: spec["criteria"][k] for k in finalists})
        if len(finalists) > J.MAX_OPTIONS:
            final_p, more = self._wide(state, fin_spec)
        else:
            ps, more = self._probs([J.from_systemone(state, fin_spec)])
            final_p = dict(zip(finalists, ps[0]))
        rest = [k for k in keys if k not in set(finalists)]
        mass = sum(first_p[k] for k in rest) or 1.0
        probs = {k: final_p[k] * (1 - self.residual) for k in finalists}
        probs.update({k: self.residual * first_p[k] / mass for k in rest})
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()}, tokens + more


    @staticmethod
    def _answer(row, p, question):
        ids = [o["id"] for o in row["options"]]
        if row["type"] == "noul":
            return {"type": "noul", "noul": p[ids.index("true")]}
        if row["type"] == "score":
            levels = question.get("criteria") or []
            levels = [levels[k] for k in J.score_keys(levels)] if isinstance(levels, dict) else list(levels)
            return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": score_confidence(p),
                    "legend": {str(i): (levels[i] if i < len(levels) else str(i)) for i in range(len(p))},
                    "probabilities": {str(i): v for i, v in enumerate(p)}}
        dist = dict(zip(ids, p))
        best = max(dist, key=dist.get)
        return {"type": "choice", "choice": best, "confidence": choice_confidence(p), "probabilities": dist}

    def _split(self, state, questions):
        answers, rows, tokens = {}, [], 0
        for k, q in questions.items():
            t = q.get("type", "choice")
            crit = q.get("criteria") or {}
            if t == "choice" and isinstance(crit, dict) and len(crit) == 1:
                only = next(iter(crit))
                answers[k] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
            elif t == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS:
                p, n = self._wide(state, q)
                tokens += n
                best = max(p, key=p.get)
                answers[k] = {"type": "choice", "choice": best, "confidence": choice_confidence(list(p.values())),
                              "probabilities": p}
            else:
                rows.append((k, J.from_systemone(state, q)))
        return answers, rows, tokens

    def decide(self, state, questions, images=None):
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]
        try:
            if images:
                self._media = self._encode_images(images)
                state = self._media_state(state, len(images))
            answers, rows, tokens = self._split(state, questions)
            if rows:
                probs, n = self._probs([r for _, r in rows])
                tokens += n
                for (k, row), p in zip(rows, probs):
                    answers[k] = self._answer(row, p, questions[k])
        finally:
            self._kept = self._media = None
        return answers, {"input_tokens": tokens, "output_tokens": 0}

    def decide_batch(self, requests):
        parts, flat = [], []
        for state, questions, *images in requests:
            if images and images[0]:
                parts.append((self.decide(state, questions, images[0])[0], [], questions))
                continue
            answers, rows, _ = self._split(state, questions)
            parts.append((answers, rows, questions))
            flat.extend(r for _, r in rows)
        graphs, self.graphs = self.graphs, {}
        try:
            probs, _ = self._probs(flat) if flat else ([], 0)
        finally:
            self.graphs, self._kept = graphs, None
        out, i = [], 0
        for answers, rows, questions in parts:
            for k, row in rows:
                answers[k] = self._answer(row, probs[i], questions[k])
                i += 1
            out.append(answers)
        return out
