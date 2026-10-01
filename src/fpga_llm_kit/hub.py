"""Read a model's config and tensor list from a local folder or the Hugging Face Hub.

Only metadata is read, never the weights. A safetensors file starts with an
8-byte length and a JSON header naming every tensor's dtype and shape, so two
small ranged requests describe a checkpoint of any size.
"""

from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HUB = "https://huggingface.co"


@dataclass(frozen=True)
class Tensor:
    dtype: str  # as safetensors spells it: BF16, F16, F32, ...
    shape: tuple[int, ...]


def load(source: str, revision: str = "main") -> tuple[dict, dict[str, Tensor]]:
    """(config, tensors) from one of three places.

    A Hub repo id like "Qwen/Qwen3-0.6B", a folder holding config.json and
    *.safetensors, or a .json snapshot written by `flk inspect --json`.
    """
    path = Path(source)
    if path.is_dir():
        return _load_folder(path)
    if path.suffix == ".json" and path.is_file():
        snapshot = json.loads(path.read_text())
        tensors = {n: Tensor(d, tuple(s)) for n, (d, s) in snapshot["tensors"].items()}
        return snapshot["config"], tensors
    return _load_hub(source, revision)


def snapshot(config: dict, tensors: dict[str, Tensor]) -> str:
    """The metadata as JSON, one tensor per line, for `load` to read back offline."""
    lines = [f"  {json.dumps(n)}: {json.dumps([t.dtype, list(t.shape)])}" for n, t in tensors.items()]
    return '{\n"config": ' + json.dumps(config) + ',\n"tensors": {\n' + ",\n".join(lines) + "\n}}\n"


def parse_header(raw: bytes) -> dict[str, Tensor]:
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return {name: Tensor(t["dtype"], tuple(t["shape"])) for name, t in header.items()}


def _load_folder(folder: Path) -> tuple[dict, dict[str, Tensor]]:
    config = json.loads((folder / "config.json").read_text())
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors file in {folder}")
    tensors: dict[str, Tensor] = {}
    for file in files:
        with file.open("rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            tensors |= parse_header(f.read(length))
    return config, tensors


def _load_hub(repo: str, revision: str) -> tuple[dict, dict[str, Tensor]]:
    def url(name: str) -> str:
        return f"{HUB}/{repo}/resolve/{revision}/{name}"

    config = json.loads(_get(url("config.json")))
    try:
        files = ["model.safetensors"]
        _get(url(files[0]), 0, 8)
    except FileNotFoundError:
        # Large checkpoints are sharded, with an index naming the shards.
        index = json.loads(_get(url("model.safetensors.index.json")))
        files = sorted(set(index["weight_map"].values()))
    tensors: dict[str, Tensor] = {}
    for name in files:
        (length,) = struct.unpack("<Q", _get(url(name), 0, 8))
        tensors |= parse_header(_get(url(name), 8, 8 + length))
    return config, tensors


def _get(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    """GET a URL, or only bytes [start, stop) of it."""
    request = urllib.request.Request(url, headers={"User-Agent": "fpga-llm-kit"})
    if start is not None:
        request.add_header("Range", f"bytes={start}-{stop - 1}")
    if token := os.environ.get("HF_TOKEN"):
        # Unredirected: the Hub redirects weights to a CDN that rejects a foreign token.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(url) from None
        if e.code in (401, 403):
            if e.headers.get("X-Error-Code") == "GatedRepo":
                raise PermissionError(
                    f"{url}: the repo is gated. Accept its licence on huggingface.co and set "
                    "HF_TOKEN, or pass a local folder."
                ) from None
            # The Hub answers 401 for a repo that does not exist, so as not to reveal private ones.
            raise FileNotFoundError(
                f"{url}: no such repo, or a private one. Check the id, and set HF_TOKEN if it "
                "is private."
            ) from None
        raise
