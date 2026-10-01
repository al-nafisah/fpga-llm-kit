"""What the hardware needs to know about a model, read from its config and tensor list.

Llama, Mistral, Qwen2, Qwen2.5, Qwen3 and SmolLM are one architecture with
different numbers. Every layer computes

    h = norm(x)
    x = x + o_proj(attention(rope(q_proj(h)), rope(k_proj(h)), v_proj(h)))
    h = norm(x)
    x = x + down_proj(act(gate_proj(h)) * up_proj(h))

and after the last layer, logits = lm_head(norm(x)). What differs between the
families is sizes, constants, and three optional extras: biases on some
projections (Qwen2), a norm on every query and key head (Qwen3), and an lm_head
that reuses the embedding table. All of that is data, so one hardware build runs
any of these models that fits within its maximum sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .hub import Tensor

FAMILIES = ("llama", "mistral", "qwen2", "qwen3")
ACTIVATIONS = ("silu", "gelu", "gelu_pytorch_tanh")
ROPE_TYPES = ("default", "linear", "llama3")

# The seven matrices in every layer, and the block each sits in.
PROJECTIONS = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}

DTYPE_BYTES = {"F64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "I16": 2,
               "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}


class Unsupported(ValueError):
    """The model needs something this hardware does not do."""


@dataclass(frozen=True)
class Matrix:
    name: str   # tensor name; "{i}" stands for the layer index
    rows: int   # outputs
    cols: int   # inputs
    count: int  # 1, or one per layer


@dataclass(frozen=True)
class Vector:
    name: str
    size: int
    count: int
    kind: str  # "norm" or "bias"


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    layers: int
    width: int                # hidden size: the length of x
    heads: int                # query heads
    kv_heads: int             # key and value heads, shared by groups of query heads
    head_dim: int
    ffn: int                  # intermediate size of the MLP
    vocab: int
    max_context: int          # longest context the model was trained for
    window: int | None        # sliding-window span, if the model uses one
    tied: bool                # lm_head reuses the embedding table
    biased: tuple[str, ...]   # projections that add a bias, e.g. q_proj, k_proj, v_proj
    qk_norm: bool             # a norm on every query and key head (Qwen3)
    activation: str           # silu (SwiGLU) or gelu (GeGLU)
    rope_theta: float
    rope_scaling: dict | None
    norm_eps: float
    checkpoint_bytes: int     # the weights as the checkpoint stores them

    @classmethod
    def from_checkpoint(cls, name: str, config: dict, tensors: dict[str, Tensor]) -> Model:
        family = config.get("model_type")
        if family not in FAMILIES:
            raise Unsupported(f"model_type {family!r}: supported are {', '.join(FAMILIES)}")
        if config.get("num_experts") or config.get("num_local_experts"):
            raise Unsupported("mixture of experts is not supported")
        if any(n.endswith((".qweight", ".scales", ".weight_scale")) for n in tensors):
            raise Unsupported("this checkpoint is already quantized: use the bf16 or fp16 original")

        activation = config.get("hidden_act", "silu")
        if activation not in ACTIVATIONS:
            raise Unsupported(f"activation {activation!r}: supported are {', '.join(ACTIVATIONS)}")

        rope_scaling = config.get("rope_scaling") or None
        if rope_scaling:
            kind = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if kind not in ROPE_TYPES:
                raise Unsupported(f"rope scaling {kind!r}: supported are {', '.join(ROPE_TYPES)}")

        # Qwen configs carry a sliding_window that is off unless use_sliding_window says so.
        window = config.get("sliding_window")
        if family.startswith("qwen") and not config.get("use_sliding_window"):
            window = None

        width = config["hidden_size"]
        heads = config["num_attention_heads"]
        model = cls(
            name=name,
            family=family,
            layers=config["num_hidden_layers"],
            width=width,
            heads=heads,
            kv_heads=config.get("num_key_value_heads", heads),
            head_dim=config.get("head_dim") or width // heads,
            ffn=config["intermediate_size"],
            vocab=config["vocab_size"],
            max_context=config["max_position_embeddings"],
            window=window,
            tied=config.get("tie_word_embeddings", False),
            biased=tuple(p for p in PROJECTIONS if any(n.endswith(f".{p}.bias") for n in tensors)),
            qk_norm=any(n.endswith(".self_attn.q_norm.weight") for n in tensors),
            activation=activation,
            rope_theta=float(config.get("rope_theta", 10000.0)),
            rope_scaling=rope_scaling,
            norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            checkpoint_bytes=sum(prod(t.shape) * DTYPE_BYTES[t.dtype] for t in tensors.values()),
        )
        model._check_tensors(tensors)
        return model

    @property
    def usable_context(self) -> int:
        """Longest context the hardware computes exactly as the model defines it.

        It does full attention; with a sliding window that is the same thing
        only while the context is no longer than the window.
        """
        return min(self.max_context, self.window or self.max_context)

    def matrices(self) -> list[Matrix]:
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        shapes = {
            "q_proj": (q, self.width),
            "k_proj": (kv, self.width),
            "v_proj": (kv, self.width),
            "o_proj": (self.width, q),
            "gate_proj": (self.ffn, self.width),
            "up_proj": (self.ffn, self.width),
            "down_proj": (self.width, self.ffn),
        }
        out = [Matrix(f"model.layers.{{i}}.{PROJECTIONS[p]}.{p}.weight", r, c, self.layers)
               for p, (r, c) in shapes.items()]
        out.append(self.embedding)
        if not self.tied:
            out.append(Matrix("lm_head.weight", self.vocab, self.width, 1))
        return out

    @property
    def embedding(self) -> Matrix:
        return Matrix("model.embed_tokens.weight", self.vocab, self.width, 1)

    @property
    def lm_head(self) -> Matrix:
        return self.embedding if self.tied else Matrix("lm_head.weight", self.vocab, self.width, 1)

    def layer_matrices(self) -> list[Matrix]:
        return [m for m in self.matrices() if "{i}" in m.name]

    def vectors(self) -> list[Vector]:
        layer = "model.layers.{i}"
        out = [
            Vector(f"{layer}.input_layernorm.weight", self.width, self.layers, "norm"),
            Vector(f"{layer}.post_attention_layernorm.weight", self.width, self.layers, "norm"),
            Vector("model.norm.weight", self.width, 1, "norm"),
        ]
        if self.qk_norm:
            out += [Vector(f"{layer}.self_attn.{n}.weight", self.head_dim, self.layers, "norm")
                    for n in ("q_norm", "k_norm")]
        rows = {m.name: m.rows for m in self.layer_matrices()}
        for p in self.biased:
            weight = f"{layer}.{PROJECTIONS[p]}.{p}.weight"
            out.append(Vector(weight.replace(".weight", ".bias"), rows[weight], self.layers, "bias"))
        return out

    @property
    def params(self) -> int:
        return (sum(m.rows * m.cols * m.count for m in self.matrices())
                + sum(v.size * v.count for v in self.vectors()))

    def expected_tensors(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for m in self.matrices():
            for i in range(m.count):
                shapes[m.name.format(i=i)] = (m.rows, m.cols)
        for v in self.vectors():
            for i in range(v.count):
                shapes[v.name.format(i=i)] = (v.size,)
        return shapes

    def _check_tensors(self, tensors: dict[str, Tensor]) -> None:
        """Refuse a checkpoint whose tensors are not exactly the ones this model implies.

        That is what catches an architecture that looks supported from its
        config but computes something else.
        """
        present = {n: t.shape for n, t in tensors.items() if not n.endswith("rotary_emb.inv_freq")}
        if self.tied:
            present.pop("lm_head.weight", None)  # some checkpoints store the shared table twice
        expected = self.expected_tensors()
        problems = (
            [f"missing {n}" for n in sorted(expected.keys() - present.keys())]
            + [f"unexpected {n}" for n in sorted(present.keys() - expected.keys())]
            + [f"{n} is {list(present[n])}, expected {list(expected[n])}"
               for n in sorted(expected.keys() & present.keys()) if tuple(present[n]) != expected[n]]
        )
        if problems:
            shown = "\n  ".join(problems[:6])
            more = f"\n  ... and {len(problems) - 6} more" if len(problems) > 6 else ""
            raise Unsupported(f"{self.name}: the tensors do not match a {self.family} model:\n  {shown}{more}")
