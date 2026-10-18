from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import limite_mlx.generate as generation


class Tokenizer:
    eos_token_ids = {31}

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"

    def encode(self, text, **kwargs):
        assert text == "rendered" and kwargs == {"add_special_tokens": False}
        return [9, 10]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "".join(
            {1: "<think>", 2: "42", 3: "</think>", 4: "ST", 5: "OP"}[i] for i in ids
        )


MODEL = SimpleNamespace(
    args=SimpleNamespace(max_position_embeddings=128, vocab_size=32)
)


def steps(monkeypatch, tokens):
    def fake(prompt, model, **kwargs):
        for token in tokens[: kwargs["max_tokens"]]:
            yield token, None

    monkeypatch.setattr(generation, "generate_step", fake)


def test_eos_metadata_and_reasoning_text(monkeypatch):
    steps(monkeypatch, [1, 2, 3, 31])
    tokenizer = Tokenizer()
    result = generation.generate(
        MODEL,
        tokenizer,
        "problem",
        max_tokens=8,
        chat_template="custom",
        chat_template_kwargs={"thinking": True},
    )
    assert (
        result.text,
        result.finish_reason,
        result.stop_reason,
        result.completion_tokens,
    ) == (
        "<think>42</think>",
        "stop",
        31,
        4,
    )
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": "problem"}],
            {
                "chat_template": "custom",
                "tokenize": False,
                "add_generation_prompt": True,
                "thinking": True,
            },
        )
    ]


def test_stop_string_spanning_tokens_and_token_limit(monkeypatch):
    steps(monkeypatch, [2, 4, 5])
    result = generation.generate(MODEL, Tokenizer(), "p", max_tokens=3, stop=("STOP",))
    assert (
        result.text,
        result.finish_reason,
        result.stop_reason,
        result.completion_tokens,
    ) == ("42", "stop", "STOP", 3)
    steps(monkeypatch, [1, 2])
    result = generation.generate(MODEL, Tokenizer(), "p", max_tokens=2)
    assert (result.text, result.finish_reason, result.stop_reason) == (
        "<think>42",
        "length",
        None,
    )


def test_context_budget_is_checked_before_generation(monkeypatch):
    monkeypatch.setattr(
        generation, "generate_step", lambda *a, **kw: pytest.fail("Must not generate")
    )
    with pytest.raises(ValueError, match="exceeds context"):
        generation.generate(MODEL, Tokenizer(), "p", max_tokens=127)


@pytest.mark.parametrize("temperature", [0.1, 0.001])
def test_temperature_precedes_nucleus_filtering(monkeypatch, temperature):
    observed = []

    def categorical(logits, *, key):
        observed.append(np.array(logits))
        return mx.argmax(logits, axis=-1)

    def fake(prompt, model, **kwargs):
        kwargs["sampler"](mx.log(mx.array([[0.6, 0.4]])))
        yield 31, None

    monkeypatch.setattr(generation.mx.random, "categorical", categorical)
    monkeypatch.setattr(generation, "generate_step", fake)
    generation.generate(
        MODEL, Tokenizer(), "p", max_tokens=1, temperature=temperature, top_p=0.7
    )
    assert np.isfinite(observed[0][0, 0]) and np.isneginf(observed[0][0, 1])


def test_sample_uses_its_own_random_state_even_after_a_stop(monkeypatch):
    def fake(prompt, model, **kwargs):
        mx.eval(kwargs["sampler"](mx.zeros((1, 32))))
        yield 31, None

    monkeypatch.setattr(generation, "generate_step", fake)
    mx.random.seed(73)
    expected = np.array(mx.random.uniform(shape=(4,)))
    mx.random.seed(73)
    generation.generate(MODEL, Tokenizer(), "p", max_tokens=1, seed=11)
    actual = np.array(mx.random.uniform(shape=(4,)))
    np.testing.assert_array_equal(actual, expected)


def test_penalties_use_full_history_and_presence_excludes_prompt(monkeypatch):
    observed = []

    def fake(prompt, model, **kwargs):
        logits = mx.ones((1, 32))
        for processor in kwargs["logits_processors"]:
            logits = processor(mx.array([9, 10, 2, 2]), logits)
        observed.append(np.array(logits))
        yield 31, None

    monkeypatch.setattr(generation, "generate_step", fake)
    generation.generate(
        MODEL, Tokenizer(), "p", max_tokens=8,
        repetition_penalty=2, presence_penalty=0.3,
    )
    assert observed[0][0, 9] == pytest.approx(0.5)
    assert observed[0][0, 2] == pytest.approx(0.2)
    assert observed[0][0, 1] == 1
