"""Single-sequence generation shared by the CLI and evaluator."""

import argparse
import json
import secrets
import sys
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    make_presence_penalty,
    make_repetition_penalty,
)

from limite_mlx.load import DEFAULT_MODEL, load


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    stop_reason: str | int | None
    completion_tokens: int


def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens=1024,
    max_model_len=None,
    temperature=0.6,
    top_p=0.95,
    top_k=None,
    min_p=None,
    presence_penalty=None,
    repetition_penalty=None,
    seed=0,
    chat_template=None,
    chat_template_kwargs=None,
    stop=(),
    stop_token_ids=None,
) -> Completion:
    if max_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require max_tokens > 0, temperature >= 0, and 0 < top_p <= 1")
    if min_p is not None and not 0 <= min_p <= 1:
        raise ValueError("min_p must be between zero and one")
    if (
        top_k is not None
        and top_k not in (-1, 0)
        and not 0 < top_k < model.args.vocab_size
    ):
        raise ValueError("top_k must be -1, 0, or smaller than the vocabulary")
    if any(not s for s in stop):
        raise ValueError("Stop strings must be nonempty")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    limit = (
        model.args.max_position_embeddings if max_model_len is None else max_model_len
    )
    if not 0 < limit <= model.args.max_position_embeddings:
        raise ValueError("Context limit exceeds the checkpoint's native context")
    if not token_ids or len(token_ids) + max_tokens > limit:
        raise ValueError(
            f"Prompt ({len(token_ids)}) plus generation ({max_tokens}) exceeds context {limit}"
        )
    eos = set(tokenizer.eos_token_ids if stop_token_ids is None else stop_token_ids)
    key = mx.random.key(secrets.randbits(32) if seed is None else seed)

    def sampler(logprobs):
        nonlocal key
        if temperature == 0:
            return mx.argmax(logprobs, axis=-1)
        # Match vLLM's temperature -> top-k -> top-p -> min-p order.
        logits = logprobs / temperature
        if top_k is not None and top_k > 0:
            logits = apply_top_k(logits, top_k)
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if top_p < 1:
            logits = apply_top_p(logits, top_p)
        if min_p:
            logits = apply_min_p(logits, min_p, 1)
        key, sample_key = mx.random.split(key)
        return mx.random.categorical(logits, key=sample_key)

    processors = []
    if repetition_penalty is not None:
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        processors.append(
            make_repetition_penalty(repetition_penalty, context_size=limit)
        )
    if presence_penalty:
        penalty = make_presence_penalty(presence_penalty, context_size=limit)
        processors.append(
            lambda tokens, logits: penalty(tokens[len(token_ids) :], logits)
        )

    generated = []
    steps = generate_step(
        mx.array(token_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
    )
    try:
        for count, (token, _) in enumerate(steps, 1):
            if token in eos:
                return Completion(
                    tokenizer.decode(generated, skip_special_tokens=True),
                    "stop",
                    token,
                    count,
                )
            generated.append(token)
            if stop:
                text = tokenizer.decode(generated, skip_special_tokens=True)
                matches = [
                    (text.index(s), i, s) for i, s in enumerate(stop) if s in text
                ]
                if matches:
                    position, _, matched = min(matches)
                    return Completion(text[:position], "stop", matched, count)
        return Completion(
            tokenizer.decode(generated, skip_special_tokens=True),
            "length",
            None,
            len(generated),
        )
    finally:
        steps.close()
        mx.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Limite on Apple Silicon with MLX")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    model, tokenizer = load(args.model, revision=args.revision)
    result = generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(result.text)
    print(
        json.dumps(
            {
                "finish_reason": result.finish_reason,
                "stop_reason": result.stop_reason,
                "completion_tokens": result.completion_tokens,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
