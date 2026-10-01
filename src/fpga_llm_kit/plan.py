"""Will a model fit a board, and how fast can it generate text there?

Generation makes one token at a time. For each token the hardware multiplies
the current vector by every weight matrix once, and attends over the cached key
and value of every earlier token. So per token it reads every weight and the
whole KV cache, and does one multiply-add per weight plus two per cached value
per query head. Whichever runs out first, DRAM bandwidth or multipliers, sets
the ceiling on tokens per second. Everything else is small next to those two.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf

from . import quant
from .boards import MIB, Board
from .model import Model


@dataclass(frozen=True)
class Build:
    """The hardware choices a plan depends on."""
    weight_bits: int = 4
    context: int = 2048
    macs_per_cycle: int = 32
    clock_mhz: float = 200.0
    dram_efficiency: float = 0.7  # share of peak bandwidth a burst reader sustains


@dataclass(frozen=True)
class Plan:
    model: Model
    board: Board
    build: Build
    context: int          # the build's context, capped at what the model supports
    weights: int          # bytes: every matrix, quantized
    vectors: int          # bytes: norms and biases
    kv_cache: int         # bytes at `context` tokens
    buffers: int          # bytes on chip: one token's activations and attention scores
    read_per_token: int   # bytes from DRAM for the last token of the context
    macs_per_token: int
    max_context: int      # longest context whose KV cache still fits in DRAM

    @property
    def dram(self) -> int:
        return self.weights + self.vectors + self.kv_cache

    @property
    def on_chip_only(self) -> bool:
        """Small enough to keep everything in block RAM, with no DRAM at all."""
        return self.dram + self.buffers <= self.board.on_chip_bytes

    @property
    def fits(self) -> bool:
        return self.on_chip_only or (self.dram <= self.board.dram_bytes_free
                                     and self.buffers <= self.board.on_chip_bytes)

    @property
    def tokens_per_s_memory(self) -> float:
        if self.on_chip_only:
            return inf
        bandwidth = self.board.dram_peak_gbps * 1e9 * self.build.dram_efficiency
        return bandwidth / self.read_per_token

    @property
    def tokens_per_s_compute(self) -> float:
        return self.build.macs_per_cycle * self.build.clock_mhz * 1e6 / self.macs_per_token

    @property
    def tokens_per_s(self) -> float:
        return min(self.tokens_per_s_memory, self.tokens_per_s_compute)

    @property
    def balanced_macs_per_cycle(self) -> int | None:
        """MACs per cycle at which the multipliers keep up with DRAM."""
        if self.on_chip_only:
            return None
        return ceil(self.macs_per_token * self.tokens_per_s_memory / (self.build.clock_mhz * 1e6))


def plan(model: Model, board: Board, build: Build = Build()) -> Plan:
    if build.weight_bits not in (4, 8):
        raise ValueError("weight_bits must be 4 or 8")
    bits = build.weight_bits
    context = min(build.context, model.usable_context)

    def size(m) -> int:
        return quant.matrix_bytes(m.rows, m.cols, bits) * m.count

    weights = sum(size(m) for m in model.matrices())
    vectors = sum(v.size * v.count * (quant.NORM_BYTES if v.kind == "norm" else quant.BIAS_BYTES)
                  for v in model.vectors())
    kv_token = quant.kv_bytes_per_token(model.layers, model.kv_heads, model.head_dim)

    # A token reads every layer matrix, the lm_head, and one row of the embedding.
    layers = sum(size(m) for m in model.layer_matrices())
    read = (layers + size(model.lm_head) + quant.matrix_bytes(1, model.width, bits)
            + vectors + kv_token * context)
    macs = (sum(m.rows * m.cols * m.count for m in model.layer_matrices())
            + model.vocab * model.width
            + 2 * model.layers * model.heads * model.head_dim * context)

    room = board.dram_bytes_free - weights - vectors
    return Plan(
        model=model, board=board, build=build, context=context,
        weights=weights, vectors=vectors, kv_cache=kv_token * context,
        buffers=_buffers(model, context), read_per_token=read, macs_per_token=macs,
        max_context=max(0, min(model.usable_context, room // kv_token)),
    )


def _buffers(model: Model, context: int) -> int:
    """On-chip RAM for one token's activations: an estimate until the RTL fixes it."""
    widest = max(model.width, model.ffn, model.heads * model.head_dim)
    return (model.width * 4       # x, the residual stream, 32-bit
            + widest              # the int8 vector entering a matrix
            + 2 * widest * 4      # 32-bit outputs: gate and up are live together
            + context * 4)        # one head's attention scores


def report(p: Plan) -> str:
    m, b, u = p.model, p.board, p.build

    def mib(n: float) -> str:
        return f"{n / MIB:,.1f}" if n >= MIB else f"{n / MIB:.2f}"

    def verdict(ok: bool) -> str:
        return "fits" if ok else "DOES NOT FIT"

    extras = [f"{m.activation} gate"]
    if m.tied:
        extras.append("lm_head shares the embedding")
    if m.biased:
        extras.append("biases on " + ", ".join(m.biased))
    if m.qk_norm:
        extras.append("q and k norms")
    if m.window:
        extras.append(f"sliding window {m.window}")
    capped = f" (capped from {u.context})" if p.context < u.context else ""

    lines = [
        f"Model   {m.name}: {m.family}, {m.params / 1e6:,.1f} M parameters, "
        f"{mib(m.checkpoint_bytes)} MiB as stored",
        f"        {m.layers} layers, width {m.width}, FFN {m.ffn}, vocabulary {m.vocab:,}",
        f"        {m.heads} query heads and {m.kv_heads} KV heads of {m.head_dim}; " + "; ".join(extras),
        f"Board   {b.name} ({b.part})",
        f"        {b.dram_mib:,} MiB {b.dram_type} at {b.dram_peak_gbps:.1f} GB/s peak, "
        f"{mib(b.on_chip_bytes)} MiB on chip, {b.dsps:,} DSPs",
        f"Build   int{u.weight_bits} weights in groups of {quant.GROUP}, int8 KV cache, "
        f"context {p.context}{capped}",
        f"        {u.macs_per_cycle} multiply-adds per cycle at {u.clock_mhz:g} MHz, "
        f"DRAM at {u.dram_efficiency:.0%} of peak",
        "",
        "Memory                         MiB",
        f"  weights                {mib(p.weights):>10}",
        f"  norms and biases       {mib(p.vectors):>10}",
        f"  KV cache               {mib(p.kv_cache):>10}   {p.context:,} tokens",
    ]
    if p.on_chip_only:
        lines.append(f"  all on chip            {mib(p.dram + p.buffers):>10}   "
                     f"of {mib(b.on_chip_bytes)}, fits: no DRAM needed")
    else:
        lines += [
            f"  in DRAM                {mib(p.dram):>10}   of {mib(b.dram_bytes_free)} free, "
            f"{verdict(p.dram <= b.dram_bytes_free)}",
            f"  on chip, estimated     {mib(p.buffers):>10}   of {mib(b.on_chip_bytes)}, "
            f"{verdict(p.buffers <= b.on_chip_bytes)}",
        ]
    if p.weights + p.vectors > b.dram_bytes_free:
        lines.append("  the weights alone overflow the free DRAM; no context fits")
    else:
        limit = ", the model's own limit" if p.max_context == m.usable_context else ""
        lines.append(f"  longest context that fits: {p.max_context:,} tokens{limit}")

    lines += ["", f"Per token, at token {p.context:,}"]
    if not p.on_chip_only:
        lines.append(f"  read from DRAM  {mib(p.read_per_token):>10} MiB   "
                     f"at most {p.tokens_per_s_memory:,.1f} tokens/s")
    lines.append(f"  multiply-adds   {p.macs_per_token / 1e6:>10,.1f} M     "
                 f"at most {p.tokens_per_s_compute:,.1f} tokens/s")
    if p.tokens_per_s_compute < p.tokens_per_s_memory:
        lines.append(f"  compute-bound: {p.balanced_macs_per_cycle:,} multiply-adds per cycle "
                     f"would reach the DRAM limit ({b.dsps:,} DSPs on this board)")
    elif u.weight_bits == 8:
        lines.append("  memory-bound: int4 weights would read about half as much")
    else:
        lines.append("  memory-bound: DRAM bandwidth sets the ceiling")
    return "\n".join(lines)
