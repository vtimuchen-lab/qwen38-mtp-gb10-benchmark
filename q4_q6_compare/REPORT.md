# Qwen3.8-27B Q4 vs Q6 with MTP7

- Date: 2026-08-17T04:43:45Z
- Hardware: NVIDIA GB10
- Context allocation: 69632 tokens
- KV cache: Q8_0 K/V; all layers on GPU; flash attention on
- Sampling: temperature 0, seed 42
- Workloads: sequence, strict JSON, Python code at ~8K/~32K/~64K
- Every Q6 prompt hash was verified against the saved Q4 MTP7 run

## Overall

| Quant | Decode tok/s | Relative to Q4 | Prompt tok/s | Acceptance | Quality | Exact vs Q4 | Semantic vs Q4 | Min case tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 UD-Q4_K_XL | 33.32 | 1.00x | 609.07 | 93.55% | 9/9 | 9/9 | 9/9 | 25.27 |
| Q6 UD-Q6_K_XL | 28.47 | 0.85x | 552.45 | 94.49% | 9/9 | 8/9 | 9/9 | 20.85 |

## By context

| Quant | Context | Decode tok/s | Prompt tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Q4 | 8k | 38.20 | 714.11 | 92.77% |
| Q4 | 32k | 33.72 | 646.30 | 94.02% |
| Q4 | 64k | 29.34 | 581.89 | 93.84% |
| Q6 | 8k | 31.63 | 671.19 | 93.64% |
| Q6 | 32k | 29.05 | 556.28 | 95.21% |
| Q6 | 64k | 25.51 | 538.94 | 94.61% |

## Comparison

- Q6 decode retention vs Q4: **85.47%**.
- Q6 prompt-processing retention vs Q4: **90.70%**.
- Q6 objective quality: **9/9**.
- Q6 semantic matches with Q4: **9/9**.
