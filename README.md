# fpga-llm-kit

Fit a pretrained language model into an FPGA with limited memory, and prove the RTL computes
it exactly.

Under construction. The [issues](https://github.com/al-nafisah/fpga-llm-kit/issues) are the
plan, in order.

## Workflow

| Step | Answers | Command |
|---|---|---|
| 1. Plan | Does the model fit the board, and how fast can it run? | `flk plan <model> --board <board>` |
| 2. Quantize | What exactly must the hardware compute? | `flk export <model>` |
| 3. Verify | Does each RTL block match that, bit for bit? | `uv run pytest tb` |
| 4. Run | Does a whole token come out right, and then text? | `flk generate <model> --prompt <text>` |

Llama, Mistral, Qwen2, Qwen2.5, Qwen3 and SmolLM compute the same layer, so the model is data:
integer weights, a descriptor of its sizes, and a few tables. One RTL build runs every model that
fits within its maximum sizes. [docs/design.md](docs/design.md) has the number format and why.

## Layout

| Path | What |
|---|---|
| `src/fpga_llm_kit/` | Python: model import, planner, quantizer, bit-exact reference |
| `rtl/` | SystemVerilog, vendor-neutral |
| `tb/` | cocotb testbenches, one per RTL module, run under Verilator |
| `tests/` | Python unit tests |
| `docs/` | the design |

## License

MIT
