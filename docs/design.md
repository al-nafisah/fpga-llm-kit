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
| an lm_head that reuses the embedding table | both point at one table |
| 1/sqrt(head dim) and other constant scales | folded into the weights at export |

The RTL is sized by maxima fixed at build time: width, FFN, head dim and context. A model runs
on a build when it fits under all of them, so a build sized for Llama-3.2-1B (width 2048) also
runs Qwen2.5-0.5B (896) and SmolLM2-135M (576). New RTL is needed only for a new operation:
mixture of experts, sliding-window attention past the window, GPT-2's LayerNorm and learned
positions, or a model that is not a transformer.

## Number format

Each row of a weight matrix is cut into groups of 32 inputs. A weight is a signed 4- or 8-bit
code, each group has an unsigned 8-bit sub-scale, and each row one scale kept as a 16-bit
mantissa and an exponent:

```
weight = code * sub_scale * row_mantissa * 2^row_exponent
```

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

## What runs where

The whole layer runs in RTL. A sequencer reads the descriptor and drives the blocks: norm,
matrix-vector engine, RoPE, attention and activation. The host loads the memory image once
(weights, descriptor, tables). Then, for each token, it writes a token id and a position and
reads back the next token id.
