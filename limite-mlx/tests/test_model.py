from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten

from limite_mlx.model import Attention, Model, ModelArgs, Mudd, Rotary, rms_norm


def args(**changes):
    base = ModelArgs(
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=24,
        vocab_size=32,
        max_position_embeddings=2048,
        attention_softmax_scale=0.1,
        sliding_window=4,
        global_window=-1,
        global_layers=[1],
        global_nope=True,
        rope_n_pairs=1,
        rope_base_local=1024.0,
        ve_dim=2,
        ve_stored_heads=2,
        ve_layers=[0, 2],
        ve_gate_channels=3,
        ve_gate_scale=2.0,
        attn_gate_channels=4,
        attn_gate_scale=2.0,
        xsa=True,
        xsa_layers=[0, 1, 2],
        xsa_normalize_eps=1e-4,
        mudd_layers=[2],
        mudd_taps=3,
        mudd_inter=3,
        mudd_tap_idx={"2": [0, 1, 2]},
        mudd_mlp=True,
        mlp_type="swiglu",
        final_softcap=0.0,
        softcap_logits={"kind": "sigmoid", "a": 23.0, "b": 5.0, "c": 7.5},
        lm_head_precision_mode="oracle_exact",
    )
    return replace(base, **changes)


def array(x):
    return np.array(x.astype(mx.float32))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_norm_matches_torch_dtype_epsilon(dtype):
    x = mx.array([[0.001, 0.02, -0.1, 0.3]]).astype(dtype)
    td = torch.bfloat16 if dtype == mx.bfloat16 else torch.float32
    reference = (
        F.rms_norm(torch.tensor(array(x), dtype=td), (4,), eps=torch.finfo(td).eps)
        .float()
        .numpy()
    )
    np.testing.assert_allclose(
        array(rms_norm(x)), reference, atol=0.008 if dtype == mx.bfloat16 else 1e-6
    )


def test_partial_rotary_uses_the_declared_sign_and_bf16_tables():
    a = args(head_dim=8, hidden_size=32, rope_n_pairs=2)
    x = mx.arange(32, dtype=mx.float32).reshape(1, 2, 2, 8) / 10
    frequencies = (1 / 1024.0) ** torch.linspace(0, 1, 2)
    frequencies = torch.cat([frequencies.repeat_interleave(2), torch.zeros(4)])
    theta = torch.arange(17, 19)[:, None] * frequencies
    cos, sin = theta.cos().bfloat16().float(), theta.sin().bfloat16().float()
    sin[:, 1::2] *= -1
    tx = torch.tensor(array(x))
    ref = cos * tx + sin * tx.reshape(1, 2, 2, 4, 2).flip(-1).reshape(tx.shape)
    out = array(Rotary(a)(x, 17))
    np.testing.assert_allclose(out, ref.numpy(), atol=2e-6)
    np.testing.assert_array_equal(out[..., 4:], array(x)[..., 4:])


def test_attention_matches_independent_torch_calculation():
    a = args(global_layers=[0], global_nope=True)
    layer = Attention(a, 0)
    mx.random.seed(7)
    layer.xsa_alpha = mx.array([0.1, -0.2, 0.3, -0.4])
    layer.ve_gate = mx.random.normal(layer.ve_gate.shape) * 0.1
    layer.attn_gate = mx.random.normal(layer.attn_gate.shape) * 0.1
    layer.qkv_scale, layer.o_scale = mx.array(0.7), mx.array(1.3)
    h = mx.random.normal((1, 7, a.hidden_size))
    ve = mx.random.normal((1, 7, a.ve_stored_heads * a.ve_dim))
    weights = {k: torch.tensor(array(v)) for k, v in tree_flatten(layer.parameters())}
    th, tve = torch.tensor(array(h)), torch.tensor(array(ve))
    q, k, v = [
        F.linear(th, weights[f"{name}_proj.weight"] * weights["qkv_scale"]).reshape(
            1, 7, heads, 4
        )
        for name, heads in (("q", 4), ("k", 2), ("v", 2))
    ]
    gate = 2 * torch.sigmoid(F.linear(th[..., :3], weights["ve_gate"]))
    v = v + gate[..., None] * F.pad(tve.reshape(1, 7, 2, 2), (0, 2))
    q, k = F.rms_norm(q, (4,)), F.rms_norm(k, (4,))
    vv = v.transpose(1, 2).repeat_interleave(2, dim=1)
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2).repeat_interleave(2, dim=1),
        vv,
        is_causal=True,
        scale=0.1,
    )
    vn = F.normalize(vv, dim=-1, eps=1e-4)
    y = (
        y
        - weights["xsa_alpha"].tanh()[None, :, None, None]
        * (y * vn).sum(-1, keepdim=True)
        * vn
    )
    y = (
        y.transpose(1, 2)
        * (2 * torch.sigmoid(F.linear(th[..., :4], weights["attn_gate"])))[..., None]
    )
    ref = F.linear(y.reshape(1, 7, 16), weights["o_proj.weight"] * weights["o_scale"])
    np.testing.assert_allclose(
        array(layer(h, ve, Rotary(a))), ref.numpy(), atol=2e-6, rtol=2e-5
    )


def test_mudd_both_paths_match_torch_with_ordered_bf16_accumulation():
    mixer = Mudd(args())
    mx.random.seed(19)
    mixer.update(
        {k: mx.random.normal(v.shape) * 0.2 for k, v in mixer.parameters().items()}
    )
    values = [mx.random.normal((1, 4, 16)).astype(mx.bfloat16) for _ in range(3)]
    w = {k: torch.tensor(array(v)) for k, v in tree_flatten(mixer.parameters())}
    ts = [torch.tensor(array(v), dtype=torch.bfloat16) for v in values]
    inner = F.gelu(F.linear(F.rms_norm(ts[-1], (16,)), w["dense1"].bfloat16()))
    for suffix in ("", "_mlp"):
        coefficients = (
            F.linear(inner, w["dense2" + suffix][2].bfloat16())
            + w["bias" + suffix][2].bfloat16()
        )
        ref = coefficients[..., :1] * ts[0]
        for i in range(1, 3):
            ref = ref + coefficients[..., i : i + 1] * ts[i]
        np.testing.assert_allclose(
            array(mixer(values, values[-1], 2, r_way=bool(suffix))),
            ref.float().numpy(),
            atol=0.008,
        )


@pytest.mark.parametrize("chunks", [[1] * 13, [3, 7, 3], [8, 1, 4]])
def test_full_model_cache_agrees_across_window_and_mudd(chunks):
    mx.random.seed(11)
    model = Model(args())
    # Exercise both residual mixer paths with meaningful nonzero weights.
    model.model.mudd.bias = mx.ones((3, 3)) / 3
    model.model.mudd.bias_mlp = mx.ones((3, 3)) / 3
    ids = mx.array([[i % 32 for i in range(13)]])
    expected = model(ids)
    cache, offset, outputs = model.make_cache(), 0, []
    for size in chunks:
        outputs.append(model(ids[:, offset : offset + size], cache))
        offset += size
        mx.eval(outputs[-1])
    np.testing.assert_allclose(
        array(mx.concatenate(outputs, axis=1)), array(expected), atol=4e-6
    )
    assert cache[0].size() <= 5
    assert cache[1].size() == 13


def test_real_1025_key_window_boundary():
    a = args(sliding_window=1024)
    layer = Attention(a, 0)
    h = mx.zeros((1, 1026, 16))
    # Identical q/k yield a uniform average; the oldest key must disappear at 1025.
    layer.q_proj.weight = mx.zeros_like(layer.q_proj.weight)
    layer.k_proj.weight = mx.zeros_like(layer.k_proj.weight)
    layer.v_proj.weight = mx.zeros_like(layer.v_proj.weight)
    layer.o_proj.weight = mx.eye(16)
    ve = mx.zeros((1, 1026, 4))
    ve[:, 0, :] = 1
    out = array(layer(h, ve, Rotary(a)))
    assert out[0, 1024, 0] == pytest.approx(1 / 1025, rel=1e-5)
    assert out[0, 1025, 0] == 0


def test_shared_head_transform_and_differentiable_scales():
    model = Model(args())
    model.model.mudd.bias = mx.ones((3, 3)) / 3
    model.model.mudd.bias_mlp = mx.ones((3, 3)) / 3
    tokens = mx.array([[1, 2, 3]])
    hidden = model.model(tokens)
    raw = hidden @ model.model.embed_tokens.weight.T
    np.testing.assert_allclose(
        array(model(tokens)), array(23 * mx.sigmoid((raw + 5) / 7.5))
    )
    assert not any(
        name.startswith("lm_head") for name, _ in tree_flatten(model.parameters())
    )
    loss, grads = nn.value_and_grad(model, lambda m: m(tokens)[0, -1, 4])(model)
    mx.eval(loss, grads)
    scale_grad = grads["model"]["layers"][0]["self_attn"]["qkv_scale"].item()
    assert np.isfinite(scale_grad) and scale_grad != 0


def test_config_rejects_inconsistent_mudd():
    with pytest.raises(ValueError, match="disagree"):
        args(mudd_tap_idx={})
