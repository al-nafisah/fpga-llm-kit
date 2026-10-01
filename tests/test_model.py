from dataclasses import replace

import pytest

from fpga_llm_kit.model import Unsupported, Model


# Every parameter the model description implies, against the checkpoint's own count.
@pytest.mark.parametrize("name, params", [
    ("SmolLM2-135M", 134_515_008),
    ("Qwen2.5-0.5B", 494_032_768),
    ("Qwen3-0.6B", 596_049_920),
    ("Llama-3.2-1B", 1_235_814_400),
    ("Mistral-7B-v0.3", 7_248_023_552),
])
def test_parameter_count_matches_the_checkpoint(load_model, name, params):
    assert load_model(name).params == params


def test_family_extras_are_read_from_the_tensors(load_model):
    qwen2, qwen3, smol, mistral = (load_model(n) for n in
                                   ("Qwen2.5-0.5B", "Qwen3-0.6B", "SmolLM2-135M", "Mistral-7B-v0.3"))
    assert qwen2.biased == ("q_proj", "k_proj", "v_proj") and not qwen2.qk_norm
    assert qwen3.qk_norm and qwen3.head_dim == 128 and not qwen3.biased
    assert smol.tied and not smol.biased and not smol.qk_norm
    assert not mistral.tied and mistral.lm_head.name == "lm_head.weight"


def test_qwen_sliding_window_counts_only_when_switched_on(snapshot):
    config, tensors = snapshot("Qwen2.5-0.5B")
    assert config["sliding_window"] and not config["use_sliding_window"]
    assert Model.from_checkpoint("q", config, tensors).window is None
    on = Model.from_checkpoint("q", {**config, "use_sliding_window": True, "sliding_window": 4096}, tensors)
    assert on.usable_context == 4096


def test_sliding_window_caps_the_usable_context(load_model):
    mistral = load_model("Mistral-7B-v0.3")
    assert mistral.usable_context == 32768
    assert replace(mistral, window=4096).usable_context == 4096


@pytest.mark.parametrize("change, message", [
    ({"model_type": "phi3"}, "model_type 'phi3'"),
    ({"num_local_experts": 8}, "mixture of experts"),
    ({"hidden_act": "relu"}, "activation 'relu'"),
    ({"rope_scaling": {"rope_type": "longrope"}}, "rope scaling 'longrope'"),
])
def test_refuses_what_the_hardware_does_not_do(snapshot, change, message):
    config, tensors = snapshot("SmolLM2-135M")
    with pytest.raises(Unsupported, match=message):
        Model.from_checkpoint("m", {**config, **change}, tensors)


def test_refuses_a_checkpoint_missing_a_tensor(snapshot):
    config, tensors = snapshot("SmolLM2-135M")
    del tensors["model.layers.7.mlp.up_proj.weight"]
    with pytest.raises(Unsupported, match="missing model.layers.7.mlp.up_proj.weight"):
        Model.from_checkpoint("m", config, tensors)


def test_refuses_a_checkpoint_with_an_unexpected_tensor(snapshot):
    config, tensors = snapshot("SmolLM2-135M")
    tensors["model.layers.3.mlp.extra_proj.weight"] = tensors["model.layers.3.mlp.up_proj.weight"]
    with pytest.raises(Unsupported, match="unexpected model.layers.3.mlp.extra_proj.weight"):
        Model.from_checkpoint("m", config, tensors)


def test_refuses_a_tensor_of_the_wrong_shape(snapshot):
    config, tensors = snapshot("SmolLM2-135M")
    config = {**config, "intermediate_size": 1024}
    with pytest.raises(Unsupported, match=r"gate_proj.weight is \[1536, 576\], expected \[1024, 576\]"):
        Model.from_checkpoint("m", config, tensors)


def test_refuses_an_already_quantized_checkpoint(snapshot):
    config, tensors = snapshot("SmolLM2-135M")
    tensors["model.layers.0.self_attn.q_proj.qweight"] = tensors.pop("model.layers.0.self_attn.q_proj.weight")
    with pytest.raises(Unsupported, match="already quantized"):
        Model.from_checkpoint("m", config, tensors)
