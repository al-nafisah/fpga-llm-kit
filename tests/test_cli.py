from fpga_llm_kit.cli import main

from conftest import MODELS


def test_plan_prints_a_report(capsys):
    assert main(["plan", str(MODELS / "SmolLM2-135M.json"), "--board", "kv260"]) == 0
    out = capsys.readouterr().out
    assert "SmolLM2-135M" in out and "tokens/s" in out


def test_inspect_describes_the_model(capsys):
    assert main(["inspect", str(MODELS / "Qwen2.5-0.5B.json")]) == 0
    assert "biases on q_proj, k_proj, v_proj" in capsys.readouterr().out


def test_an_unsupported_model_is_an_error_not_a_traceback(tmp_path, capsys):
    bad = (MODELS / "SmolLM2-135M.json").read_text().replace('"llama"', '"gpt2"', 1)
    (tmp_path / "bad.json").write_text(bad)
    assert main(["inspect", str(tmp_path / "bad.json")]) == 1
    assert "model_type 'gpt2'" in capsys.readouterr().err


def test_boards_lists_them(capsys):
    assert main(["boards"]) == 0
    assert "kv260" in capsys.readouterr().out
