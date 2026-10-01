from dataclasses import replace
from math import inf

import pytest

from fpga_llm_kit import quant
from fpga_llm_kit.boards import MIB, Board
from fpga_llm_kit.plan import Build, plan

# A board with round numbers, so the expected values can be worked by hand.
BOARD = Board(key="test", name="Test board", part="X", dsps=1000,
              bram_kbits=8192, uram_kbits=0, dram_type="DDR4", dram_mib=1024,
              dram_bus_bits=64, dram_mts=2000, reserved_mib=0)  # 16 GB/s peak


def test_matrix_bytes_by_hand():
    # 576 inputs: 18 groups. int4 is 288 code bytes, int8 576, plus 18 sub-scales
    # and a 4-byte row scale.
    assert quant.matrix_bytes(576, 576, 4) == 576 * (288 + 18 + 4)
    assert quant.matrix_bytes(576, 576, 8) == 576 * (576 + 18 + 4)
    with pytest.raises(ValueError):
        quant.matrix_bytes(10, 100, 4)


def test_smollm2_by_hand(load_model):
    model = load_model("SmolLM2-135M")
    p = plan(model, BOARD, Build(weight_bits=4, context=1000))

    # One key and one value of 64 int8 plus a scale byte, for 3 KV heads in 30 layers.
    kv_token = 30 * 2 * 3 * (64 + 1)
    assert p.kv_cache == kv_token * 1000
    # Per layer: q and o are 576x576, k and v 192x576, gate and up 1536x576, down 576x1536.
    layer_macs = 2 * 576 * 576 + 2 * 192 * 576 + 3 * 1536 * 576
    attention_macs = 2 * 30 * 9 * 64 * 1000
    assert p.macs_per_token == 30 * layer_macs + 49152 * 576 + attention_macs
    # The tied embedding is stored once, and read whole for the lm_head every token.
    embedding = quant.matrix_bytes(49152, 576, 4)
    assert p.weights == sum(quant.matrix_bytes(m.rows, m.cols, 4) * m.count
                            for m in model.layer_matrices()) + embedding
    assert p.read_per_token == (p.weights + quant.matrix_bytes(1, 576, 4)
                                + p.vectors + p.kv_cache)


def test_the_slower_of_dram_and_multipliers_sets_the_ceiling(load_model):
    model = load_model("SmolLM2-135M")
    slow_dram = plan(model, replace(BOARD, dram_mts=100), Build(macs_per_cycle=4096))
    few_macs = plan(model, BOARD, Build(macs_per_cycle=8))

    assert slow_dram.tokens_per_s == slow_dram.tokens_per_s_memory < slow_dram.tokens_per_s_compute
    assert few_macs.tokens_per_s == few_macs.tokens_per_s_compute < few_macs.tokens_per_s_memory
    # At the balanced width the multipliers just keep up with DRAM.
    balanced = plan(model, BOARD, Build(macs_per_cycle=few_macs.balanced_macs_per_cycle))
    assert balanced.tokens_per_s_compute >= balanced.tokens_per_s_memory
    assert balanced.tokens_per_s_compute < balanced.tokens_per_s_memory * 1.01


def test_longest_context_is_what_the_free_dram_leaves_for_the_kv_cache(load_model):
    model = load_model("SmolLM2-135M")
    p = plan(model, BOARD)
    kv_token = quant.kv_bytes_per_token(model.layers, model.kv_heads, model.head_dim)
    exactly_1000 = (p.weights + p.vectors + 1000 * kv_token + MIB - 1) // MIB
    tight = plan(model, replace(BOARD, dram_mib=exactly_1000))
    assert 1000 <= tight.max_context < 1000 + MIB // kv_token + 1
    assert plan(model, replace(BOARD, reserved_mib=BOARD.dram_mib)).max_context == 0


def test_does_not_fit_when_the_weights_alone_overflow(load_model):
    mistral = load_model("Mistral-7B-v0.3")
    p = plan(mistral, BOARD)
    assert p.weights > BOARD.dram_bytes_free and not p.fits


def test_context_is_capped_at_what_the_model_supports(load_model):
    model = replace(load_model("SmolLM2-135M"), window=512)
    assert plan(model, BOARD, Build(context=4096)).context == 512


def test_a_small_enough_model_needs_no_dram(load_model):
    tiny = replace(load_model("SmolLM2-135M"), layers=2, width=64, heads=2, kv_heads=1,
                   head_dim=32, ffn=128, vocab=512)
    p = plan(tiny, replace(BOARD, bram_kbits=32 * 1024), Build(context=256))
    assert p.on_chip_only and p.fits
    assert p.tokens_per_s_memory == inf and p.tokens_per_s == p.tokens_per_s_compute


def test_weight_bits_must_be_4_or_8(load_model):
    with pytest.raises(ValueError):
        plan(load_model("SmolLM2-135M"), BOARD, Build(weight_bits=3))
