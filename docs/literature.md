# Literature

What published work says about compressing a language model for a small FPGA, and what this
project takes from it. Reviewed on 2026-10-01. Each number names the table or figure it comes
from. Perplexities compare only within one paper, because context length moves them: Llama-2-7B
scores 5.47 on WikiText-2 at 2,048 tokens and 5.12 at 4,096.

## Conclusions

1. **Quantize first.** At the same size, quantization keeps more accuracy than pruning, and the
   gap is widest on small models. On Llama-3.2-1B-Instruct, perplexity on English text goes
   from 16.2 to 17.5 with int4 AWQ and to 28.0 with 50% SparseGPT ([Zhou et al.], Table 6).
   [Kuzmin et al.] reach the same verdict across many networks, and [LLM-KICK] on knowledge
   tasks.
2. **Calibrate the rounding.** Meta's own int4 Llama-3.2-1B uses 4-bit weights in groups of 32
   and int8 activations with one scale per token, nearly the format here. On the instruct
   model, MMLU is 49.3 in bf16, 43.3 with plain post-training quantization, 47.3 with SpinQuant
   and GPTQ, and 49.0 with quantization-aware training ([Llama 3.2 model card]). Every
   calibrated method below runs at export and writes the same integers, so none needs RTL.
3. **The models here are the hard ones.** Quantization damage grows with training tokens and
   shrinks with model size ([Ouyang et al.]). SmolLM2-135M saw 2T tokens, 15,000 per parameter
   ([SmolLM2], Sec. 6). Qwen3-0.6B at 4 bits in groups of 128 loses 5.1 MMLU points even with
   GPTQ, while its 8-bit weights lose nothing ([Zheng et al.], Tables 1 and 3).
4. **The 4-bit weights cost the most; int8 activations cost little once outliers are spread.**
   A few channels carry activations far larger than the rest ([LLM.int8()], [SmoothQuant]
   Fig. 4), and one scale per vector then leaves the others few levels. On Llama-3.2-1B with
   int4 weights and int8 activations and KV cache, WikiText-2 perplexity is 13.4 in bf16, 20.7
   with plain rounding, 17.3 with GPTQ, 15.3 with rotations folded into the weights, and 14.3
   with an online Hadamard transform added before down_proj. SmoothQuant, which moves the
   activation outliers into the weights, takes it to about 100 ([SpinQuant], Table 1). So
   scaling has to protect the 4-bit weights, as [AWQ] does, and rotation is the stronger fix.
   No paper compares these methods at symmetric int4 in groups of 32 on a model of 1B or less;
   that is the first thing to measure here.
5. **Power-of-two scales are enough for int8 vectors; keep the KV cache at int8.** int8 in
   blocks of 32 with a power-of-two scale moves Llama-3.2-1B from 9.06 to 9.08 perplexity
   ([INT v.s. FP], Table 15). This design uses one scale for a whole vector, where outliers
   stretch the range further, so it needs measuring. A 4-bit KV cache is nearly free on Llama-2
   after rotation (5.47 to 5.51, [QuaRot] Table 6), but plain int4 takes Llama-3-8B from 5.54 to
   7.89 ([KVQuant], Table 17).
6. **2:4 sparsity buys about 1.2x here, not 2x.** Each kept weight needs a 2-bit position, so
   int4 2:4 costs 3.25 bits per weight against 4.25 dense, the same as dense int3. The tied
   lm_head is 21 to 28% of these models' weights and is not pruned. A token then reads 17 to 19%
   fewer weight bytes. One-shot 2:4 also hurts small models most: SparseGPT 2:4 multiplies
   WikiText-2 perplexity by 3.4 on a Llama-3 1B and by 2.0 on Llama-2-7B ([Thanos], Table 2).
7. **2:4 and rotation do not mix.** A rotation mixes columns, so it destroys a 2:4 pattern, and
   pruning in the rotated basis fails: SparseGPT at 40% on Llama-3-8B goes from 8.5 to 98.2
   perplexity once QuaRot is applied ([Kim et al.], Table 4). A 2:4 model can take per-channel
   scales, which keep zeros zero, but not a rotation.
8. **Structured pruning needs distillation on billions of tokens.** Without it, 20% pruning
   costs 1 to 7 zero-shot points on LLaMA-7B depending on the method, and every method collapses
   at 50% ([LLM-Pruner] Table 1, [FLAP] Tables 1 and 2). Qwen models lose MMLU once 20% of
   their layers are dropped ([Gromov et al.]). With distillation it works: Minitron-8B, pruned
   from a 15B model, beats a model of its size trained from scratch on 40 times the tokens
   ([Minitron], Table 2). Llama-3.2-1B is itself pruned and distilled from Llama 3.1. A model
   pruned to the same sizes in every layer runs on the descriptor as is.
9. **Perplexity and averages hide damage.** Pruned models lose knowledge tasks at 25 to 35%
   sparsity while perplexity holds to 45 to 60% ([LLM-KICK], Figs. 1 and 2). Quantized models
   change 5% or more of their answers while average accuracy moves under 2 points
   ([Dutta et al.], Fig. 1). Report each task, the fraction of answers that flip, and the KL
   divergence to the bf16 model.
10. **FPGA papers measure speed, rarely the accuracy of what they compute.** Of 21 designs
    reviewed, 9 report no accuracy, most of the rest report perplexity from a software model,
    and none checks the hardware against an integer reference end to end or runs a standard
    benchmark suite on the hardware's arithmetic. That is the gap this project fills.
11. **On an edge board, decode waits on DRAM.** Weight fetch dominates decode latency on a
    ZCU102 ([MEADOW], Fig. 1). The best KV260 designs reach 84 to 94% of its 19.2 GB/s on 7B and
    8B int4 models ([Li et al.], [Hummingbird]). None reviewed comes near that on a 100M to 1B
    model, and designs that move the lm_head or nonlinear operations to the ARM cores no longer
    measure the model they ship.

## Quantization

WikiText-2 perplexity at 2,048 tokens, or MMLU where marked.

| Model | Setting | bf16 | Plain rounding | Calibrated | Source |
|---|---|---|---|---|---|
| Llama-2-7B | W4, groups of 128 | 5.47 | 5.72 | GPTQ 5.61, AWQ 5.62, OmniQuant 5.58 | [OmniQuant] T1 |
| Llama-2-7B | W3, groups of 128 | 5.47 | 6.66 | GPTQ 6.29, AWQ 6.24, OmniQuant 6.03 | [OmniQuant] T1 |
| Llama-3-8B | W4, groups of 128 | 6.1 | 8.5 | GPTQ 6.5, AWQ 6.6 | [Huang et al.] T1 |
| Qwen3-0.6B | W4, one scale per row; ppl / MMLU | 12.7 / 52.3 | 24.0 / 35.9 | GPTQ 18.2 / 40.4, AWQ 15.6 / 47.3 | [Zheng et al.] T1 |
| Qwen3-0.6B | W4, groups of 128; ppl / MMLU | 12.7 / 52.3 | | GPTQ 14.9 / 47.2, AWQ 16.6 / 43.8 | [Zheng et al.] T3 |
| Qwen3-0.6B | W8A8; ppl / MMLU | 12.7 / 52.3 | | SmoothQuant 13.0 / 51.7 | [Zheng et al.] T1 |
| Llama-3.2-1B-Instruct | W4 in groups of 32, A8 per token; MMLU | 49.3 | 43.3 | SpinQuant + GPTQ 47.3, QAT 49.0 | [Llama 3.2 model card] |
| Llama-3.2-1B | W4, A8, KV8; ppl / 8-task mean | 13.4 / 56.9 | 20.7 / 55.7 | GPTQ 17.3 / 54.8, SmoothQuant about 100 / 47.1, folded rotations + GPTQ 15.3 / 55.7, plus online Hadamard 14.3 / 55.8 | [SpinQuant] T1 |
| Llama-3.2-1B | W8A8 in blocks of 32, power-of-two scales | 9.06 | 9.08 | | [INT v.s. FP] T15 |
| Llama-2-7B | W8, A8 and KV8, after rotation | 5.47 | 5.50 | | [QuaRot] T3 |
| Llama-2-7B | W4A4, KV cache 4-bit | 5.47 | | QuaRot + GPTQ 6.10 | [QuaRot] T1 |
| Llama-2-7B | KV cache 4-bit only, after rotation | 5.47 | 5.51 | | [QuaRot] T6 |
| Llama-3-8B | KV cache 4-bit, one scale per token | 5.54 | 7.89 | | [KVQuant] T17 |

SpinQuant calibrates its rotations and GPTQ on WikiText-2 train, so its WikiText-2 numbers are
not zero-shot; the 8-task mean is.

What each method does, and what it costs here:

- [GPTQ] quantizes a matrix one column at a time and spreads each column's rounding error over
  the columns not yet quantized, weighted by the inverse Hessian of the layer's inputs. It
  needs one Hessian per input size, at most 8,192 x 8,192 for Llama-3.2-1B's down_proj, which
  numpy handles.
- [AWQ] and [SmoothQuant] scale each input channel, activations down and weights up, by a
  factor found on calibration text. Both fold into the RMSNorm weight before the matrix, or
  into the rows of the matrix before it (v_proj for o_proj, up_proj for down_proj). AWQ picks
  the factor that minimizes the layer's output error, so it protects the weights; SmoothQuant
  splits the range evenly between activations and weights, which breaks 4-bit weights.
- [QuaRot] multiplies the residual stream by a randomized Hadamard matrix and [SpinQuant] by a
  learned rotation. SpinQuant's training needs a GPU; QuaRot's rotation does not. Both fold
  every norm weight into the next matrix first, so the lm_head no longer equals the embedding
  table and needs its own copy in DRAM; it reads no more bytes per token. The online Hadamard
  before down_proj is additions and subtractions: the FFN sizes here are a power of two, alone
  or times 12 or 76 (1,536, 3,072, 4,864, 8,192), and Hadamard matrices of those orders exist.
- Embedding and lm_head: Meta keeps both at 8 bits per channel in its int4 Llama 3.2.
- Small groups help small models: GPTQ on Qwen3-0.6B goes from 18.2 perplexity with one scale
  per row to 14.9 with groups of 128 ([Zheng et al.], T1 and T3). This project uses groups of
  32.

## Pruning

One-shot pruning, no retraining. WikiText-2 perplexity, then the mean of the zero-shot tasks
each paper uses.

| Model | Dense | 50%: SparseGPT / Wanda | 2:4: SparseGPT / Wanda | Source |
|---|---|---|---|---|
| OPT-125M | 27.66 | 37.07 / 38.96 | | [Wanda] T10 |
| a Llama-3 1B | 9.75 / 52.9 | 18.8 / 46.8, 23.4 / 44.7 | 32.7 / 42.6, 78.6 / 37.6 | [Thanos] T2, T3 |
| TinyLlama-1.1B | 7.97 / 51.3 | 11.1 / 47.7, 11.5 / 46.9 | 19.2 / 43.1, 27.2 / 40.9 | [Thanos] T2, T3 |
| Llama-2-7B, 4,096 tokens | 5.12 / 59.7 | 6.51 / 56.2, 6.42 / 56.2 | 10.17 / 50.9, 11.02 / 48.8 | [Wanda] T2, T3 |
| Llama-3-8B, 4,096 tokens | 5.76 | | 17.64 / 23.40; MaskLLM 8.50 | [MaskLLM] T12 |

- Learned 2:4 masks close much of the gap but need GPUs: [MaskLLM] reaches 6.72 on Llama-2-7B
  at 2:4, against SparseGPT's 10.42, after 1,280 A100-hours. Recovery fine-tuning does too:
  2:4 LLaMA-7B goes from 11.53 to 8.24 with LoRA and 7.02 with full fine-tuning ([Wanda] T6).
- On small models SparseGPT beats Wanda, the reverse of the trend on large ones
  ([Zhou et al.]). Wanda only needs activation norms; SparseGPT is GPTQ's algorithm with a
  mask, so the two share code.
- Structured pruning results that keep every layer the same size, and so run on one
  descriptor: [Minitron], [Minitron in practice], [Sheared-LLaMA], [ShortGPT], [Gromov et al.].
  Results with a different size per layer need a per-layer descriptor: [LLM-Pruner], [FLAP].
  After distillation, width pruning beats depth pruning: Llama-3.1-8B pruned to 4B scores 60.5
  MMLU by width and 58.7 by depth ([Minitron in practice], T1).

## Pruning and quantization together

- Order matters with magnitude pruning: on Llama-2-7B at 2:4 and int8, pruning first gives
  9.37 perplexity and quantizing first 14.65 ([Harma et al.], T1). With SparseGPT or Wanda the
  order barely matters (App. F, T7). [SparseGPT] prunes and quantizes in one pass.
- On OPT models from 2.7B up, 50% sparsity plus 4 bits beats 3-bit GPTQ at the same size
  ([SparseGPT], Fig. 6). No paper we found makes that comparison at 1B or below, or for 2:4.
- Joint methods on Llama-2-7B at 2:4 plus 4 bits: SparseGPT with GPTQ gives 12.07 perplexity,
  and [SLiM] with low-rank adapters 7.77 (T10), against 5.47 dense.

## Storage of sparse weights

Bits per weight in this project's format, groups of 32 with an 8-bit sub-scale:

| Weights | int4 | int8 |
|---|---|---|
| dense | 4.25 | 8.25 |
| 2:4, a 2-bit position per kept weight ([Mishra et al.]) | 3.25 | 5.25 |
| 50% unstructured, a 1-bit mask per weight | 3.25 on average | 5.25 on average |

A mask pays for itself above 25% sparsity at int4, but the number of codes per group then
varies, and the stream turns irregular. FPGA designs with sparsity, [FlightLLM] (N:M in blocks
of 16), [EdgeLLM] and [AccLLM] (2:4), all sit on HBM boards, and each paid in accuracy: EdgeLLM's
most aggressive setting takes perplexity from 29.9 to 120.9.

## Evaluation

The protocol most papers share, and what this project runs:

- Perplexity on WikiText-2 test (`Salesforce/wikitext`, `wikitext-2-raw-v1`): join with
  `"\n\n"`, tokenize once, cut into non-overlapping 2,048-token windows, drop the rest ([GPTQ]).
  C4: the first 256 x 2,048 tokens of the validation shard `en/c4-validation.00000-of-00008`.
  lm-evaluation-harness's own `wikitext` task measures something else (word perplexity over
  rolling windows) and cannot be compared with these.
- Zero-shot accuracy with [lm-evaluation-harness]: `arc_easy`, `arc_challenge`, `hellaswag`,
  `piqa`, `winogrande`, `boolq`, `lambada_openai`, `openbookqa`. Report acc_norm where the task
  has it, acc otherwise, and keep both.
- MMLU 5-shot where the baseline is above chance: Qwen2.5-0.5B (47.5), Qwen3-0.6B (52.8),
  Llama-3.2-1B and up. SmolLM2-135M scores 31.5 even in the cloze form.
- Agreement with the bf16 model on the perplexity text: per-token KL divergence, and how often
  both pick the same next token. On each task, the fraction of answers that flip
  ([Dutta et al.]).
- Statistics: one task's standard error is 0.5 points on HellaSwag (10,042 questions) and 2
  points on OpenBookQA (500). The paired difference per question, as [Miller] recommends, is
  far tighter than comparing two scores. A change under 1 point on one task is noise unless
  the paired test says otherwise.
- Pitfalls: whether a BOS token is added (Llama-3.2 adds one, Qwen and SmolLM2 do not; older
  lm-evaluation-harness versions dropped it), sequence length, calibration on C4 train and
  then scoring C4, and chat templates on instruct models. Record each.
- Cost: lm-evaluation-harness 0.4.13 runs a custom model without torch. The 8-task suite is
  5.2M tokens and MMLU 5-shot about 9.6M. Integer matrix products in numpy run at 0.3 GOP/s,
  but float64 holds every integer below 2^53 exactly and runs at 150 GFLOP/s on a 12-core CPU.
  At that rate the suite takes about 3 hours for SmolLM2-135M and 17 for Llama-3.2-1B.

## FPGA designs

Decode at batch 1. "Accuracy" is what the paper reports and where it was measured.

| Design | Board | Model | Weights / activations | tokens/s | Accuracy |
|---|---|---|---|---|---|
| [DFX], MICRO 2022 | 4x U280 | GPT-2 1.5B | fp16 | 72.7 | 3 tasks, on the hardware |
| [FlightLLM], FPGA 2024 | U280 | LLaMA2-7B | 3 to 5 bits, N:M / int8 | about 55 | perplexity, platform not stated |
| [Chen et al.], TRETS 2024 | U280 | GPT-2 355M | int8 / int8 | plotted only | LAMBADA, in PyTorch |
| [EdgeLLM], TCAS-I 2025 | VCU128 | ChatGLM2-6B | int4, sparse / fp16 | 85.8 | perplexity and 4 tasks, platform not stated |
| [LlamaF], WF-IoT 2024 | ZCU102 | TinyLlama 1.1B | int8 / int8 | 1.48 | perplexity |
| [MEADOW], MLSys 2025 | ZCU102 | OPT-1.3B | int8 / int8 | about 2 | LAMBADA, in software |
| [Li et al.], DATE 2025 | KV260 | LLaMA2-7B | AWQ int4 / fp16 | 4.9 | none |
| [Hummingbird], ICCAD 2025 | KV260 | LLaMA3-8B | GPTQ int4 / fp16 | 4.8 | none |
| [On-device Qwen2.5], 2025 | KV260 | Qwen2.5-0.5B | AWQ int4 / fp32 | 5.1, co-simulated | WNLI only |
| [TeLLMe], 2025 | KV260 | BitNet 0.73B | ternary / int8 | 24.6 to 11.2 | perplexity |

Surveys: [Li et al. 2024] place FPGA decode at 3.6 to 450 tokens/s and warn that some papers
"offer estimates without hardware validation". [Blankestijn et al.] find accuracy reporting
too inconsistent to compare designs.

## References

The [README](../README.md) reproduces one figure from each paper marked CC BY 4.0, under that
license.

- [AccLLM] Liang et al. AccLLM: Accelerating Long-Context LLM Inference Via Algorithm-Hardware Co-Design. 2025.
- [AWQ] Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024.
- [Blankestijn et al.] Blankestijn, Odyurt, Yousefzadeh. Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey. Journal of Systems Architecture, 2026.
- [Chen et al.] Chen et al. Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference. ACM TRETS, 2024.
- [DFX] Hong et al. DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based Text Generation. MICRO 2022.
- [Dutta et al.] Dutta et al. Accuracy is Not All You Need. NeurIPS 2024. CC BY 4.0.
- [EdgeLLM] Huang et al. EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models. IEEE TCAS-I, 2025.
- [FLAP] An et al. Fluctuation-based Adaptive Structured Pruning for Large Language Models. AAAI 2024.
- [FlightLLM] Zeng et al. FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs. FPGA 2024.
- [GPTQ] Frantar et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. CC BY 4.0.
- [Gromov et al.] Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers. ICLR 2025.
- [Harma et al.] Harma et al. Effective Interplay between Sparsity and Quantization: From Theory to Practice. ICLR 2025.
- [Huang et al.] Huang et al. An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs. Visual Intelligence, 2024.
- [Hummingbird] Li et al. Hummingbird: A Smaller and Faster Large Language Model Accelerator on Embedded FPGA. ICCAD 2025.
- [INT v.s. FP] Chen et al. INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats. 2025.
- [Kim et al.] Kim et al. Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression. ICLR 2026.
- [Kuzmin et al.] Kuzmin et al. Pruning vs Quantization: Which is Better? NeurIPS 2023.
- [KVQuant] Hooper et al. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. NeurIPS 2024.
- [Li et al.] Li et al. Pushing up to the Limit of Memory Bandwidth and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA. DATE 2025.
- [Li et al. 2024] Li et al. Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective. 2024.
- [Llama 3.2 model card] Meta. Llama 3.2 model card, quantization section. 2024.
- [LlamaF] Xu et al. LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs. WF-IoT 2024.
- [LLM-KICK] Jaiswal et al. Compressing LLMs: The Truth is Rarely Pure and Never Simple. ICLR 2024.
- [LLM-Pruner] Ma et al. LLM-Pruner: On the Structural Pruning of Large Language Models. NeurIPS 2023.
- [LLM.int8()] Dettmers et al. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 2022.
- [lm-evaluation-harness] EleutherAI. lm-evaluation-harness, version 0.4.13.
- [MaskLLM] Fang et al. MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models. NeurIPS 2024.
- [MEADOW] Moitra et al. MEADOW: Memory-efficient Dataflow and Data Packing for Low Power Edge LLMs. MLSys 2025. CC BY 4.0.
- [Miller] Miller. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. 2024.
- [Minitron] Muralidharan et al. Compact Language Models via Pruning and Knowledge Distillation. NeurIPS 2024. CC BY 4.0.
- [Minitron in practice] Sreenivas et al. LLM Pruning and Distillation in Practice: The Minitron Approach. 2024.
- [Mishra et al.] Mishra et al. Accelerating Sparse Deep Neural Networks. 2021.
- [OmniQuant] Shao et al. OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models. ICLR 2024.
- [On-device Qwen2.5] Xiang et al. On-Device Qwen2.5: Efficient LLM Inference with Model Compression and Hardware Acceleration. 2025.
- [Ouyang et al.] Ouyang et al. Low-Bit Quantization Favors Undertrained LLMs. ACL 2025.
- [QuaRot] Ashkboos et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024.
- [Sheared-LLaMA] Xia et al. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning. ICLR 2024.
- [ShortGPT] Men et al. ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. Findings of ACL 2025.
- [SLiM] Mozaffari et al. SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression. ICML 2025.
- [SmolLM2] Ben Allal et al. SmolLM2: When Smol Goes Big. 2025.
- [SmoothQuant] Xiao et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. CC BY 4.0.
- [SparseGPT] Frantar and Alistarh. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. ICML 2023. CC BY 4.0.
- [SpinQuant] Liu et al. SpinQuant: LLM Quantization with Learned Rotations. ICLR 2025.
- [TeLLMe] Qiao et al. TeLLMe v2: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs. 2025.
- [Thanos] Ilin and Richtarik. Thanos: A Block-wise Pruning Algorithm for Efficient Large Language Model Compression. 2025.
- [Wanda] Sun et al. A Simple and Effective Pruning Approach for Large Language Models. ICLR 2024.
- [Zheng et al.] Zheng et al. An Empirical Study of Qwen3 Quantization. 2025.
- [Zhou et al.] Zhou, Kurz, Zhao. Revisiting Pruning vs Quantization for Small Language Models. Findings of EMNLP 2025.

[AccLLM]: https://arxiv.org/abs/2505.03745
[AWQ]: https://arxiv.org/abs/2306.00978
[Blankestijn et al.]: https://arxiv.org/abs/2609.01212
[Chen et al.]: https://arxiv.org/abs/2312.15159
[DFX]: https://arxiv.org/abs/2209.10797
[Dutta et al.]: https://arxiv.org/abs/2407.09141
[EdgeLLM]: https://arxiv.org/abs/2407.21325
[FLAP]: https://arxiv.org/abs/2312.11983
[FlightLLM]: https://arxiv.org/abs/2401.03868
[GPTQ]: https://arxiv.org/abs/2210.17323
[Gromov et al.]: https://arxiv.org/abs/2403.17887
[Harma et al.]: https://arxiv.org/abs/2405.20935
[Huang et al.]: https://arxiv.org/abs/2404.14047
[Hummingbird]: https://arxiv.org/abs/2507.03308
[INT v.s. FP]: https://arxiv.org/abs/2510.25602
[Kim et al.]: https://arxiv.org/abs/2603.18426
[Kuzmin et al.]: https://arxiv.org/abs/2307.02973
[KVQuant]: https://arxiv.org/abs/2401.18079
[Li et al.]: https://arxiv.org/abs/2502.10659
[Li et al. 2024]: https://arxiv.org/abs/2410.04466
[Llama 3.2 model card]: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
[LlamaF]: https://arxiv.org/abs/2409.11424
[LLM-KICK]: https://arxiv.org/abs/2310.01382
[LLM-Pruner]: https://arxiv.org/abs/2305.11627
[LLM.int8()]: https://arxiv.org/abs/2208.07339
[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
[MaskLLM]: https://arxiv.org/abs/2409.17481
[MEADOW]: https://arxiv.org/abs/2503.11663
[Miller]: https://arxiv.org/abs/2411.00640
[Minitron]: https://arxiv.org/abs/2407.14679
[Minitron in practice]: https://arxiv.org/abs/2408.11796
[Mishra et al.]: https://arxiv.org/abs/2104.08378
[OmniQuant]: https://arxiv.org/abs/2308.13137
[On-device Qwen2.5]: https://arxiv.org/abs/2504.17376
[Ouyang et al.]: https://arxiv.org/abs/2411.17691
[QuaRot]: https://arxiv.org/abs/2404.00456
[Sheared-LLaMA]: https://arxiv.org/abs/2310.06694
[ShortGPT]: https://arxiv.org/abs/2403.03853
[SLiM]: https://arxiv.org/abs/2410.09615
[SmolLM2]: https://arxiv.org/abs/2502.02737
[SmoothQuant]: https://arxiv.org/abs/2211.10438
[SparseGPT]: https://arxiv.org/abs/2301.00774
[SpinQuant]: https://arxiv.org/abs/2405.16406
[TeLLMe]: https://arxiv.org/abs/2510.15926
[Thanos]: https://arxiv.org/abs/2504.05346
[Wanda]: https://arxiv.org/abs/2306.11695
[Zheng et al.]: https://arxiv.org/abs/2505.02214
[Zhou et al.]: https://aclanthology.org/2025.findings-emnlp.645/
