# rtl

SystemVerilog, in the synthesizable subset that both Verilator and Vivado accept. One module per
file, named after the module. It uses no vendor primitives. RAMs and multipliers are inferred,
so the same source builds for AMD, Intel or Lattice parts.

Blocks pass data over valid/ready streams and read DRAM over AXI4. Sizes that change from model
to model are run-time inputs; only maxima are parameters.
