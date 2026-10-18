from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    attention_softmax_scale: float
    sliding_window: int
    global_window: int
    global_layers: list[int]
    global_nope: bool
    rope_n_pairs: int
    rope_base_local: float
    ve_dim: int
    ve_stored_heads: int
    ve_layers: list[int]
    ve_gate_channels: int
    ve_gate_scale: float
    attn_gate_channels: int
    attn_gate_scale: float
    xsa: bool
    xsa_layers: list[int]
    xsa_normalize_eps: float
    mudd_layers: list[int]
    mudd_taps: int
    mudd_inter: int
    mudd_tap_idx: dict[str, list[int]]
    mudd_mlp: bool
    mlp_type: str
    final_softcap: float
    softcap_logits: dict
    lm_head_precision_mode: str
    model_type: str = "limite"

    @classmethod
    def from_dict(cls, config: dict) -> "ModelArgs":
        conventions = {
            "model_type": "limite",
            "tie_word_embeddings": True,
            "rms_norm_has_weight": False,
            "rms_norm_eps_mode": "torch_finfo_default",
            "qk_norm": "rms_pre_rope",
            "pos_mode": "rope",
            "rope_per_layer": False,
            "rope_style": "interleaved_pairs_odd_lane_sign_flip",
            "rope_cos_sin_dtype": "bfloat16",
            "ve_head_slice": "first_num_key_value_heads",
            "ve_applied_before_qk_norm": True,
            "attn_gate_applied": "per_head_before_o_proj",
            "mudd_accumulation": "ordered_left_to_right",
            "mudd_hist_convention": "hist[0] = rms_norm(embedding); hist[j] = output of layer j-1",
            "sliding_window_convention": (
                "k >= q - sliding_window, inclusive of the query token "
                "(span = sliding_window + 1 keys)"
            ),
        }
        for name, expected in conventions.items():
            if config.get(name) != expected:
                raise ValueError(
                    f"Unsupported {name}={config.get(name)!r}; expected {expected!r}"
                )
        if bool(config.get("mudd")) != bool(config.get("mudd_layers")):
            raise ValueError("mudd and mudd_layers disagree")
        if config.get("mudd_at") != config.get("mudd_layers"):
            raise ValueError("mudd_at and mudd_layers disagree")
        formula = {
            "swiglu": "(silu(gate_proj(h)) * up_proj(h)) @ down_proj, no clamp on either factor",
            "relu2": "relu(fc(h)) ** 2 @ down_proj",
        }.get(config.get("mlp_type"))
        if formula is None or config.get("mlp_formula") != formula:
            raise ValueError("Unsupported MLP formula")
        try:
            return cls(**{f.name: config[f.name] for f in fields(cls)})
        except KeyError as error:
            raise ValueError(
                f"Missing Limite configuration field: {error.args[0]}"
            ) from error

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Query heads must be a multiple of key/value heads")
        if not 0 < 2 * self.rope_n_pairs <= self.head_dim or self.head_dim % 2:
            raise ValueError("Invalid partial rotary dimensions")
        if (
            not 0 < self.ve_dim <= self.head_dim
            or self.ve_stored_heads < self.num_key_value_heads
        ):
            raise ValueError("Invalid value embedding dimensions")
        if not 0 <= self.attn_gate_channels <= self.hidden_size:
            raise ValueError("Invalid attention gate width")
        if not 0 < self.ve_gate_channels <= self.hidden_size:
            raise ValueError("Invalid value gate width")
        if self.mlp_type not in {"swiglu", "relu2"}:
            raise ValueError(f"Unsupported mlp_type={self.mlp_type!r}")
        if self.lm_head_precision_mode not in {"oracle_exact", "fp32_accumulate"}:
            raise ValueError("Unsupported output head precision")
        if (
            self.softcap_logits.get("kind") != "sigmoid"
            or self.softcap_logits.get("c", 0) <= 0
        ):
            raise ValueError("Expected a sigmoid output transform with positive c")
        if set(map(int, self.mudd_tap_idx)) != set(self.mudd_layers):
            raise ValueError("mudd_tap_idx and mudd_layers disagree")
        for layer, taps in self.mudd_tap_idx.items():
            if not 0 <= int(layer) < self.num_hidden_layers:
                raise ValueError("MUDD layer is out of range")
            if not 0 < len(taps) <= self.mudd_taps or any(
                t < 0 or t > int(layer) for t in taps
            ):
                raise ValueError("MUDD taps must refer to available residual states")
        for indices in (self.global_layers, self.ve_layers, self.xsa_layers):
            if any(i < 0 or i >= self.num_hidden_layers for i in indices):
                raise ValueError("Layer index is out of range")


def rms_norm(x: mx.array) -> mx.array:
    # PyTorch's omitted epsilon is finfo(input.dtype).eps, not a fixed 1e-6.
    return mx.fast.rms_norm(x, None, mx.finfo(x.dtype).eps)


def scaled_linear(x: mx.array, projection: nn.Linear, scale: mx.array) -> mx.array:
    weight = projection.weight
    return x @ (weight * scale.astype(weight.dtype)).T


class Rotary(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        frequency = (1 / args.rope_base_local) ** mx.linspace(0, 1, args.rope_n_pairs)
        self._frequency = mx.concatenate(
            [mx.repeat(frequency, 2), mx.zeros(args.head_dim - 2 * args.rope_n_pairs)]
        )

    def __call__(self, x: mx.array, offset: int) -> mx.array:
        theta = (
            mx.arange(offset, offset + x.shape[2], dtype=mx.float32)[:, None]
            * self._frequency
        )
        cosine = mx.cos(theta).astype(mx.bfloat16)
        sine = mx.sin(theta).astype(mx.bfloat16)
        sign = mx.array([1, -1] * (x.shape[-1] // 2), dtype=mx.bfloat16)
        flipped = x.reshape(*x.shape[:-1], -1, 2)[..., ::-1].reshape(x.shape)
        return cosine * x + (sine * sign) * flipped


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.args = args
        self._uses_ve = index in args.ve_layers
        self._uses_xsa = args.xsa and index in args.xsa_layers
        self._uses_rope = not (args.global_nope and index in args.global_layers)
        distance = (
            args.global_window if index in args.global_layers else args.sliding_window
        )
        self._window = None if distance < 0 else distance + 1
        kv_size = args.num_key_value_heads * args.head_dim
        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.qkv_scale = mx.array(1.0)
        self.o_scale = mx.array(1.0)
        self.xsa_alpha = mx.zeros(args.num_attention_heads)
        if self._uses_ve:
            self.ve_gate = mx.zeros((args.ve_stored_heads, args.ve_gate_channels))
        if args.attn_gate_channels:
            self.attn_gate = mx.zeros(
                (args.num_attention_heads, args.attn_gate_channels)
            )

    def __call__(self, h, value_embeds, rotary, cache=None):
        a = self.args
        batch, length, _ = h.shape
        q = scaled_linear(h, self.q_proj, self.qkv_scale).reshape(
            batch, length, a.num_attention_heads, a.head_dim
        )
        k = scaled_linear(h, self.k_proj, self.qkv_scale).reshape(
            batch, length, a.num_key_value_heads, a.head_dim
        )
        v = scaled_linear(h, self.v_proj, self.qkv_scale).reshape(k.shape)
        if self._uses_ve:
            ve = value_embeds.reshape(batch, length, a.ve_stored_heads, a.ve_dim)
            ve = mx.pad(
                ve[:, :, : a.num_key_value_heads],
                [(0, 0), (0, 0), (0, 0), (0, a.head_dim - a.ve_dim)],
            )
            gate = a.ve_gate_scale * mx.sigmoid(
                h[..., : a.ve_gate_channels]
                @ self.ve_gate[: a.num_key_value_heads].astype(h.dtype).T
            )
            v = v + gate[..., None] * ve.astype(v.dtype)
        q, k = rms_norm(q).transpose(0, 2, 1, 3), rms_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        current_values = v
        offset = 0 if cache is None else cache.offset
        if self._uses_rope:
            q, k = rotary(q, offset), rotary(k, offset)
        mask = create_attention_mask(h, cache, window_size=self._window)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=a.attention_softmax_scale, mask=mask
        )
        if self._uses_xsa:
            vn = mx.repeat(
                current_values, a.num_attention_heads // a.num_key_value_heads, axis=1
            )
            vn = vn.astype(mx.float32)
            vn = vn / mx.maximum(
                mx.linalg.norm(vn, axis=-1, keepdims=True), a.xsa_normalize_eps
            )
            projection = mx.sum(y.astype(mx.float32) * vn, axis=-1, keepdims=True)
            alpha = mx.tanh(self.xsa_alpha.astype(mx.float32))[None, :, None, None]
            y = y - (alpha * projection * vn).astype(y.dtype)
        y = y.transpose(0, 2, 1, 3)
        if a.attn_gate_channels:
            gate = a.attn_gate_scale * mx.sigmoid(
                h[..., : a.attn_gate_channels] @ self.attn_gate.astype(h.dtype).T
            )
            y = y * gate.astype(y.dtype)[..., None]
        return scaled_linear(y.reshape(batch, length, -1), self.o_proj, self.o_scale)


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._kind = args.mlp_type
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        if self._kind == "swiglu":
            self.gate_proj = nn.Linear(
                args.hidden_size, args.intermediate_size, bias=False
            )

    def __call__(self, h):
        up = self.up_proj(h)
        activated = (
            nn.silu(self.gate_proj(h)) * up
            if self._kind == "swiglu"
            else mx.square(nn.relu(up))
        )
        return self.down_proj(activated)


class DecoderLayer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.self_attn = Attention(args, index)
        self.mlp = MLP(args)
        self.resid_lambda_attn = mx.array(1.0)
        self.post_lambda_attn = mx.array(1.0)
        self.resid_lambda_mlp = mx.array(1.0)
        self.post_lambda_mlp = mx.array(1.0)

    def __call__(self, h, attn_input, residual, ve, rotary, cache):
        out = self.self_attn(attn_input, ve, rotary, cache)
        h = (
            self.resid_lambda_attn.astype(h.dtype) * residual
            + self.post_lambda_attn.astype(h.dtype) * out
        )
        out = self.mlp(rms_norm(h))
        return (
            self.resid_lambda_mlp.astype(h.dtype) * h
            + self.post_lambda_mlp.astype(h.dtype) * out
        )


class Mudd(nn.Module):
    def __init__(self, args):
        super().__init__()
        shape = (args.num_hidden_layers, args.mudd_taps)
        self.dense1 = mx.zeros((args.mudd_inter, args.hidden_size))
        self.dense2 = mx.zeros((*shape, args.mudd_inter))
        self.bias = mx.zeros(shape)
        if args.mudd_mlp:
            self.dense2_mlp = mx.zeros((*shape, args.mudd_inter))
            self.bias_mlp = mx.zeros(shape)

    def __call__(self, values, current, index, r_way=False):
        inner = nn.gelu(rms_norm(current) @ self.dense1.astype(current.dtype).T)
        dense2, bias = (
            (self.dense2_mlp, self.bias_mlp) if r_way else (self.dense2, self.bias)
        )
        weights = inner @ dense2[index, : len(values)].astype(inner.dtype).T
        weights = weights + bias[index, : len(values)].astype(weights.dtype)
        out = weights[..., :1].astype(values[0].dtype) * values[0]
        for i, value in enumerate(values[1:], 1):
            out = out + weights[..., i : i + 1].astype(value.dtype) * value
        return out


class LimiteModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.value_embeds = nn.Embedding(
            args.vocab_size, args.ve_stored_heads * args.ve_dim
        )
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.mudd = Mudd(args)
        self._rotary = Rotary(args)
        self._taps = {int(k): v for k, v in args.mudd_tap_idx.items()}
        self._retained = {i for taps in self._taps.values() for i in taps}

    def __call__(self, tokens, cache=None):
        h = rms_norm(self.embed_tokens(tokens))
        ve = self.value_embeds(tokens)
        history = {0: h}
        for i, layer in enumerate(self.layers):
            if i in self._taps:
                values = [h if tap == i else history[tap] for tap in self._taps[i]]
                attn_input = rms_norm(self.mudd(values, h, i))
                residual = (
                    self.mudd(values, h, i, r_way=True) if self.args.mudd_mlp else h
                )
            else:
                attn_input, residual = rms_norm(h), h
            h = layer(
                h,
                attn_input,
                residual,
                ve,
                self._rotary,
                None if cache is None else cache[i],
            )
            if i + 1 in self._retained:
                history[i + 1] = h
        if self.args.final_softcap > 0:
            h = self.args.final_softcap * mx.tanh(h / self.args.final_softcap)
        return rms_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LimiteModel(args)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache()
            if layer.self_attn._window is None
            else RotatingKVCache(layer.self_attn._window)
            for layer in self.layers
        ]

    def __call__(self, tokens, cache=None):
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Expected one cache per decoder layer")
        hidden = self.model(tokens, cache)
        weight = self.model.embed_tokens.weight
        if self.args.lm_head_precision_mode == "oracle_exact":
            raw = (hidden @ weight.T).astype(mx.float32)
        else:
            raw = hidden.astype(mx.float32) @ weight.astype(mx.float32).T
        cap = self.args.softcap_logits
        return cap["a"] * mx.sigmoid((raw + cap["b"]) / cap["c"])

    def sanitize(self, weights):
        weights = dict(weights)
        if "lm_head.weight" in weights:
            head = weights.pop("lm_head.weight")
            if "model.embed_tokens.weight" in weights:
                if not mx.array_equal(
                    head, weights["model.embed_tokens.weight"]
                ).item():
                    raise ValueError("Input embeddings and output head must be tied")
            else:
                weights["model.embed_tokens.weight"] = head
        return weights
