# Design

## The model is data

Llama, Mistral, Qwen2, Qwen2.5, Qwen3 and SmolLM compute the same layer:

```
h = norm(x)
x = x + o_proj(attention(rope(q_proj(h)), rope(k_proj(h)), v_proj(h)))
h = norm(x)
x = x + down_proj(act(gate_proj(h)) * up_proj(h))
```

After the last layer, `logits = lm_head(norm(x))`. Whatever differs between the families is
loaded at run time:

| What differs | Where it goes |
|---|---|
| width, layers, heads, KV heads, head dim, FFN, vocabulary | registers in a model descriptor |
| RoPE theta and scaling, SiLU or GELU | tables written at export |
| biases on q, k and v (Qwen2) | a descriptor bit and a bias vector per matrix |
| a norm on every query and key head (Qwen3) | a descriptor bit; the norm block is reused |
| an lm_head that reuses the embedding table | both point at one table, or at two once export changes the lm_head |
| 1/sqrt(head dim) and other constant scales | folded into the weights at export |
| code width and sparsity of the layers and of the lm_head | descriptor fields |

The RTL is sized by maxima fixed at build time: width, FFN, head dim and context. A model runs
on a build when it fits under all of them, so a build sized for Llama-3.2-1B (width 2048) also
runs Qwen2.5-0.5B (896) and SmolLM2-135M (576). New RTL is needed only for a new operation:
mixture of experts, sliding-window attention past the window, GPT-2's LayerNorm and learned
positions, or a model that is not a transformer.

## Compression

Decoding a token reads every weight once, so on a board the speed is DRAM bandwidth divided by
bytes per token. Compression cuts those bytes, on the host at export, by quantizing and pruning.
Each choice ends up as data in the number format below, so the RTL does not know how a model
was compressed. [literature.md](literature.md) has the evidence behind each choice.

### Quantization

Rounding each weight to the nearest code is the baseline. Calibrated methods use a calibration
text to choose better codes, and write the same integers:

- GPTQ quantizes a matrix one column at a time and moves each column's rounding error onto the
  columns not yet quantized, weighted by the inverse Hessian of the layer's inputs.
- Per-channel scaling (AWQ) divides an input channel's activations by a factor and multiplies
  the matching weight column by it, choosing the factor that minimizes the layer's output error.
  The division folds into the RMSNorm weight before the matrix, or into the rows of v_proj (for
  o_proj) and of up_proj (for down_proj).
- Rotation (QuaRot) multiplies the residual stream by an orthogonal matrix, which spreads outlier
  channels across all channels. It folds into the weights, except for a Hadamard transform on
  the input of down_proj, which would be a new RTL block of additions and subtractions. Folding
  needs the final norm weight inside the lm_head, so a rotated model stores the lm_head apart
  from the embedding table.

The export starts with rounding, GPTQ and scaling, in numpy, and keeps whichever measures best
for each model. Folded rotation comes next. The Hadamard block is built only if measurements
show down_proj's int8 input costing accuracy that folded rotation does not recover.

The layers and the lm_head can use different code widths. Meta keeps the lm_head and embedding
of its int4 Llama 3.2 at 8 bits. For the models here that costs 20 to 26% more bytes per token,
so it is measured, not assumed.

### Pruning

2:4 sparsity keeps 2 of every 4 consecutive weights in a row. A group of 32 inputs then holds 16
codes, a 2-bit position for each and its sub-scale, and needs 16 multiply-adds instead of 32.
SparseGPT chooses the mask and quantizes in the same pass as GPTQ. A rotation would spread the
zeros, so a 2:4 model uses scaling only. The lm_head stays dense, which leaves the saving at 17
to 19% of the bytes per token for these models. 2:4 at int4 costs the same 3.25 bits per weight
as dense int3, so it earns its place only by beating dense int3 on accuracy.

Structured pruning removes layers, heads or FFN channels and leaves a smaller dense model. When
every layer keeps the same sizes, the descriptor runs it unchanged. Recovering its accuracy
takes distillation over billions of tokens, which is GPU work outside this project. So the
project runs models others have pruned and distilled (Llama-3.2-1B is one), and measures layer
dropping without retraining as a baseline.

Unstructured sparsity is left out. It varies the number of codes per group, and the small
models here do not survive the 60 to 70% sparsity at which its mask would pay well.

## Number format

Each row of a weight matrix is cut into groups of 32 inputs. A weight is a signed 4- or 8-bit
code, each group has an unsigned 8-bit sub-scale, and each row one scale kept as a 16-bit
mantissa and an exponent:

```
weight = code * sub_scale * row_mantissa * 2^row_exponent
```

A 2:4 matrix keeps the same groups and scales. Each group stores only its 16 kept codes, and
for each a 2-bit position within its block of 4 inputs:

| Weights | Bytes per group of 32 | Bits per weight |
|---|---|---|
| int8 | 33 | 8.25 |
| int4 | 17 | 4.25 |
| int8, 2:4 | 21 | 5.25 |
| int4, 2:4 | 13 | 3.25 |

Activations entering a matrix are int8 with one power-of-two scale per vector, chosen so the
largest magnitude lands in 64 to 127. A matrix-vector product is then integer from end to end:
sum code times activation within a group, multiply by the group's sub-scale, add up the groups
in 48 bits, multiply by the row mantissa, and shift once by both exponents.

Between blocks, the residual stream is 32-bit fixed point. Its fraction bits are chosen per
model at export, from the largest values seen on a calibration text, and stored in the
descriptor.

The KV cache is int8 with one power-of-two scale per head vector. Norm weights are int16 and
biases int32.

Checkpoints come in as bf16, fp16 or fp32 safetensors. GGUF has about 30 quantization types,
each packed its own way, and reading them natively would need a decoder per type in the RTL. A
GGUF model is dequantized and requantized to this one format instead.

## Measuring the cost

Benchmarks run on the integer reference, which the testbenches hold equal to the RTL bit for
bit. A benchmark score is therefore the score of what the FPGA computes, with no float model
standing in for it. The bf16 model runs through the same code as the baseline.

- Perplexity on WikiText-2 test and on the C4 validation shard, in 2,048-token windows as GPTQ
  measures it, and also at the build's context when that is shorter.
- The zero-shot suite of lm-evaluation-harness: ARC-Easy, ARC-Challenge, HellaSwag, PIQA,
  WinoGrande, BoolQ, LAMBADA and OpenBookQA. MMLU 5-shot where the bf16 model is above chance.
- Agreement with bf16: per-token KL divergence and top-1 agreement on the perplexity text, and
  the share of answers that flip on each task.

Each compressed model reports its scores, the change from bf16 with a paired 95% interval, its
bytes per token, and the planner's tokens per second.

Every integer operation works on one token's vectors, so the reference can run a whole sequence
through a matrix at once and still produce the bits the RTL produces one token at a time. It
computes those products in float64 at BLAS speed: every integer in them stays below 2^36
(a code times its sub-scale times an activation is under 2^22, summed over at most 2^14
inputs), and float64 holds integers up to 2^53 exactly. lm-evaluation-harness runs without
torch, so it is an optional `eval` dependency group and the core stays numpy and tokenizers.

## What runs where

The whole layer runs in RTL. A sequencer reads the descriptor and drives the blocks: norm,
matrix-vector engine, RoPE, attention and activation. The host loads the memory image once
(weights, descriptor, tables). Then, for each token, it writes a token id and a position and
reads back the next token id.
