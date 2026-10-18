"""Load original Limite artifacts without a model registry or import hooks."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model, load_tokenizer

from limite_mlx.model import Model, ModelArgs

DEFAULT_MODEL = "paradigma-inc/limite-1b-violetto"
DEFAULT_REVISION = "b1f3d572ccacb6919f4d64c321b70ba034ddaef2"


def load(path_or_repo=DEFAULT_MODEL, *, revision=None):
    path = Path(path_or_repo).expanduser()
    if not path.is_dir():
        if revision is None and path_or_repo == DEFAULT_MODEL:
            revision = DEFAULT_REVISION
        path = Path(
            snapshot_download(
                path_or_repo,
                revision=revision,
                allow_patterns=["*.json", "*.jinja", "model*.safetensors"],
            )
        )
    config = json.loads((path / "config.json").read_text())
    if any(
        config.get(k) for k in ("quantization", "quantization_config", "model_file")
    ):
        raise ValueError("Load the original, unquantized Limite checkpoint")
    model, _ = load_model(path, get_model_classes=lambda config: (Model, ModelArgs))
    generation = json.loads((path / "generation_config.json").read_text())
    eos = generation.get("eos_token_id", config.get("eos_token_id"))
    eos = eos if isinstance(eos, list) else [eos]
    tokenizer = load_tokenizer(path, {"trust_remote_code": False}, eos_token_ids=eos)
    model.eval()
    return model, tokenizer
