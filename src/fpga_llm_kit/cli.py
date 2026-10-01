"""flk: the command line.

    flk inspect MODEL           what the hardware would need to run it
    flk plan MODEL --board B    whether it fits that board, and how fast it could run
    flk boards                  the bundled board profiles

MODEL is a Hugging Face repo id, a local folder with config.json and
*.safetensors, or a .json snapshot written by `flk inspect --json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import hub
from .boards import MIB, all_boards, board
from .model import Model, Unsupported
from .plan import Build, plan, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flk", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="check a model and describe what it needs")
    inspect.add_argument("model")
    inspect.add_argument("--revision", default="main")
    inspect.add_argument("--json", action="store_true",
                         help="print a snapshot to plan from later, offline")

    p = commands.add_parser("plan", help="fit a model to a board")
    p.add_argument("model")
    p.add_argument("--board", required=True, help="a bundled board, or a path to a .toml")
    p.add_argument("--revision", default="main")
    defaults = Build()
    p.add_argument("--weight-bits", type=int, choices=(4, 8), default=defaults.weight_bits)
    p.add_argument("--context", type=int, default=defaults.context, help="tokens of context")
    p.add_argument("--macs-per-cycle", type=int, default=defaults.macs_per_cycle)
    p.add_argument("--clock-mhz", type=float, default=defaults.clock_mhz)
    p.add_argument("--dram-efficiency", type=float, default=defaults.dram_efficiency,
                   help="share of peak DRAM bandwidth sustained (default %(default)s)")

    commands.add_parser("boards", help="list the bundled boards")

    args = parser.parse_args(argv)
    try:
        if args.command == "boards":
            for b in all_boards():
                print(f"{b.key:<12} {b.name}: {b.dram_mib:,} MiB {b.dram_type} at "
                      f"{b.dram_peak_gbps:.1f} GB/s, {b.on_chip_bytes / MIB:.1f} MiB on chip, "
                      f"{b.dsps:,} DSPs")
            return 0

        config, tensors = hub.load(args.model, args.revision)
        name = Path(args.model).stem if Path(args.model).exists() else args.model
        if args.command == "inspect" and args.json:
            # Validate first, so a snapshot is only ever written for a supported model.
            Model.from_checkpoint(name, config, tensors)
            sys.stdout.write(hub.snapshot(config, tensors))
            return 0
        model = Model.from_checkpoint(name, config, tensors)
        if args.command == "inspect":
            print(_describe(model))
            return 0

        build = Build(args.weight_bits, args.context, args.macs_per_cycle,
                      args.clock_mhz, args.dram_efficiency)
        print(report(plan(model, board(args.board), build)))
        return 0
    except (Unsupported, KeyError, FileNotFoundError, PermissionError) as e:
        print(f"flk: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1


def _describe(m: Model) -> str:
    lines = [
        f"{m.name}: a {m.family} model the hardware supports",
        f"  {m.params:,} parameters in {m.layers} layers",
        f"  width {m.width}, FFN {m.ffn} ({m.activation}), vocabulary {m.vocab:,}",
        f"  {m.heads} query heads, {m.kv_heads} KV heads, head dim {m.head_dim}",
        f"  context {m.usable_context:,}"
        + (f" (sliding window {m.window})" if m.window else ""),
        f"  RoPE theta {m.rope_theta:g}"
        + (f", {m.rope_scaling.get('rope_type', m.rope_scaling.get('type'))} scaling"
           if m.rope_scaling else ""),
        f"  lm_head {'shares the embedding' if m.tied else 'is its own matrix'}",
    ]
    if m.biased:
        lines.append(f"  biases on {', '.join(m.biased)}")
    if m.qk_norm:
        lines.append("  a norm on every query and key head")
    lines.append("  matrices (rows x cols, count):")
    for mat in m.matrices():
        lines.append(f"    {mat.name.replace('model.', '').replace('.weight', ''):<34} "
                     f"{mat.rows:>7} x {mat.cols:<6} x{mat.count}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
