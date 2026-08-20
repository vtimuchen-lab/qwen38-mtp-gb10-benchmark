# Qwen3.8 Q4 complex MTP depth validation

- Same frozen 10-case subsets for each mode: IFEval, GSM8K, tool calls, and chat.
- Sampling: temperature 0, seed 42; context 65,536; Q8_0 KV; prefix cache disabled.

| Mode | IFEval strict prompt | GSM8K | Tools | Chat valid | Decode tok/s | Acceptance | Warm RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 90% | 90% | 100% | 100% | 9.40 | — | 1704.7 |
| mtp5 | 90% | 90% | 100% | 100% | 19.64 | 47.32% | 2179.7 |
| mtp7 | 90% | 90% | 100% | 100% | 18.28 | 37.15% | 2182.7 |
| mtp11 | 90% | 90% | 100% | 100% | 15.38 | 24.29% | 2189.0 |
| mtp13 | 90% | 90% | 100% | 100% | 14.32 | 21.13% | 2192.9 |
| mtp16 | 100% | 90% | 100% | 100% | 12.67 | 16.72% | 2195.6 |

The mini-sweep is a paired regression check, not a replacement for the full 50/100-case benchmark.
