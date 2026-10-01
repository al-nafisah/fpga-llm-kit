import email.message
import json
import struct
import urllib.error

import pytest

from fpga_llm_kit import hub


def write_safetensors(path, tensors):
    """A real safetensors file: length, JSON header, then the zeroed data."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def test_reads_config_and_headers_from_a_folder(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    write_safetensors(tmp_path / "model-1.safetensors", {"a.weight": ("BF16", [4, 8], 64)})
    write_safetensors(tmp_path / "model-2.safetensors", {"b.weight": ("F32", [3], 12)})

    config, tensors = hub.load(str(tmp_path))

    assert config == {"model_type": "llama"}
    assert tensors == {"a.weight": hub.Tensor("BF16", (4, 8)), "b.weight": hub.Tensor("F32", (3,))}


def test_a_snapshot_reads_back_as_it_was_written(tmp_path):
    config = {"model_type": "qwen2", "hidden_size": 896}
    tensors = {"x.weight": hub.Tensor("BF16", (896, 896)), "x.bias": hub.Tensor("BF16", (896,))}
    path = tmp_path / "snap.json"
    path.write_text(hub.snapshot(config, tensors))

    assert hub.load(str(path)) == (config, tensors)


@pytest.mark.parametrize("error_code, raised, message", [
    ("GatedRepo", PermissionError, "the repo is gated"),
    (None, FileNotFoundError, "no such repo, or a private one"),
])
def test_a_gated_repo_and_a_missing_one_get_different_errors(monkeypatch, error_code, raised, message):
    # The Hub answers 401 to both; only the X-Error-Code header tells them apart.
    def refuse(request, timeout):
        headers = email.message.Message()
        if error_code:
            headers["X-Error-Code"] = error_code
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", headers, None)

    monkeypatch.setattr(hub.urllib.request, "urlopen", refuse)
    with pytest.raises(raised, match=message):
        hub.load("some-org/some-model")
