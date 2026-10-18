# Limite MLX

An MLX implementation of Limite, loading the original Hugging Face weights
directly on Apple Silicon. Python 3.12 and a Mac with Metal support are required.

```bash
cd limite-mlx
uv sync --locked
uv run --locked limite-generate --prompt "Solve 2x + 3 = 11."
```

The default checkpoint is Violetto at the evaluator's pinned revision. Weights
and tokenizer assets use the standard Hugging Face cache. The model is implemented
locally; no third-party Limite port or remote Python model code is loaded.

The implementation is in three modules: `model.py` defines the architecture,
`load.py` loads weights and tokenizer, and `generate.py` provides generation and
the command line. Model parameters retain their original dtypes and learned
scales remain differentiable. Full training workflows are not yet validated.

```python
from limite_mlx.load import load
from limite_mlx.generate import generate

model, tokenizer = load()
result = generate(model, tokenizer, "Solve 2x + 3 = 11.", max_tokens=1024)
print(result.text)
```

Generation uses a separate PRNG key per sample. Temperature is applied before
top-k, top-p, and min-p filtering, matching the evaluation parameter semantics.
The returned completion includes its finish reason, matched stop token or string,
and generated token count. Reasoning delimiters remain in the decoded text.

To evaluate the port, install the sibling evaluator's `mlx` extra and use
`limite-eval ... --backend mlx`; see its [README](../limite-evals/README.md).
Evaluation reuses the existing datasets, prompts, graders, journal, and reports.

```bash
uv sync --locked --group dev
uv run --locked --group dev pytest
uv run --locked --group dev ruff check src tests
```

The tests compare numerical components against PyTorch and exercise the local
attention boundary, MUDD paths, shared head, gradients, and cached decoding.
The model preserves the checkpoint's BF16-specific epsilon and rounding choices;
numerical results across frameworks need not be bitwise identical.
