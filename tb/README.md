# tb

One cocotb testbench per RTL module, `test_<module>.py`, built and run under Verilator by
cocotb's Python runner:

```bash
uv run pytest tb
```

Each test drives the RTL and the Python reference in `fpga_llm_kit` with the same inputs and
requires equal outputs, bit for bit.
