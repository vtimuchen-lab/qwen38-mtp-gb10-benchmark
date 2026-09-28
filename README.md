# Qwen3.8-27B on NVIDIA GB10: MTP7 Q4 vs Q6, depth sweep, and an MTP10 operational default

> **Status:** controlled quality/performance run complete; the 72-hour soak was stopped by the operator after 6 h 17 min and is reported only as a partial run; the local operational default is now MTP10.
> **Run date:** 2026-08-17 onward (UTC)
> **Runtime:** `llama.cpp` commit `d2f83055d6e3b379b5d34c4837122a918cf402c2`

## TL;DR

I compared the `UD-Q4_K_XL` and `UD-Q6_K_XL` variants of Qwen3.8-27B under the same `llama.cpp` MTP7 configuration on an NVIDIA GB10 system.

- Q4 showed no consistent quality disadvantage on the tested subsets.
- Q6 delivered 17.0–19.9% fewer decode tokens per second than Q4 across every suite. Across the complete workload, Q4 produced 21.48 tok/s and Q6 produced 17.43 tok/s, making Q4 23.2% faster when Q6 is used as the denominator.
- Q6 was slightly better on IFEval, while Q4 was slightly better on MMLU-Pro and HumanEval. None of the paired quality differences was statistically significant at this sample size.
- Both variants passed all 9 long-context checks, all 20 simple tool-call checks, and all 100 deterministic repetition checks.
- MTP acceptance was highly workload-dependent: about 26% on IFEval, 46% overall, 97% on structured tool calls, and 100% on the deterministic stability prompt.
- A follow-up Q4 depth sweep found that the fastest depth depends strongly on workload: MTP16 won the predictable aggregate by only 0.24% over MTP13, while MTP5 was fastest on the mixed IFEval/GSM8K/tool/chat mini-suite.
- The local Q4 and Q6 launchers now use MTP10 as an operational compromise. On the predictable sweep it reached 37.01 tok/s (4.03x baseline), retained the best measured minimum-case rate at 25.62 tok/s, passed 9/9 objective checks, and remained semantically identical to baseline on 9/9 cases. This setting was not part of the full Q4/Q6 quality comparison or the mixed mini-sweep, so those historical results remain labelled MTP7.
- In a 12-turn agent-session A/B with an 18,243-token static prefix, `cache_prompt=true` reduced prompt-processing time by 89.74% and improved end-to-end wall time by 7.13x with identical 12/12 task quality.

For this host and workload, **Q4 with MTP10 is the operational default**. Q6 may still be useful for instruction-following-sensitive workloads, but this run did not demonstrate a statistically reliable quality gain that offsets its larger file and lower throughput.

This is a controlled paired comparison, not a replacement for full public leaderboard runs.

## What was tested

Each quantization ran the same seven suites:

1. MMLU-Pro: knowledge and reasoning.
2. GSM8K: grade-school mathematics.
3. HumanEval: Python code generation with execution of official tests.
4. IFEval: strict and loose instruction-following evaluation.
5. Synthetic long-context retrieval and structured output at approximately 8K, 32K, and 60K tokens.
6. OpenAI-compatible function/tool calling.
7. One hundred identical deterministic requests for short-run stability.

Per quantization, the run contained 419 scored cases and 519 model API calls. MMLU-Pro used two calls per case, which accounts for the extra 100 calls. Across Q4 and Q6 together, the experiment produced 838 scored cases and 1,038 API calls.

## Models

The GGUF files came from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

| Variant | Quantization | File size | SHA256 |
|---|---|---:|---|
| Q4 | `UD-Q4_K_XL` | 17,923,394,624 bytes (17.92 GB) | `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372` |
| Q6 | `UD-Q6_K_XL` | 25,924,152,384 bytes (25.92 GB) | `739202186fd9389bb58497c58b56c8a0d4253d99d20131e6a0427e363e678fc8` |

The Q6 file is 44.6% larger than the Q4 file.

## Test system

| Component | Value |
|---|---|
| GPU/platform | NVIDIA GB10 |
| NVIDIA driver | `580.173.02` |
| Architecture | `aarch64` |
| Logical CPUs | 20 |
| Usable host memory reported by Linux | 121 GiB |
| Kernel | Linux `6.17.0-1029-nvidia` |
| Runtime | `llama-server` version 1, commit `d2f8305` |
| Compiler | GNU 13.3.0 |

Only one quantization was resident during a measurement run. Competing Ollama and vLLM model services were stopped, then restored after the benchmark.

## llama.cpp configuration

Both variants used exactly the same server configuration:

```text
--ctx-size 65536
--parallel 1
--gpu-layers all
--fit off
--flash-attn on
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 2048
--ubatch-size 2048
--jinja
--reasoning off
--reasoning-budget 0
--reasoning-format deepseek
--spec-type draft-mtp
--spec-draft-n-max 7
--spec-draft-ngl all
```

Every model request used:

```text
temperature = 0
seed = 42
stream = false
cache_prompt = false
```

The test therefore compares Q4 and Q6 **with MTP7 enabled**. It does not compare MTP-on against MTP-off, so MTP acceptance should not be interpreted as an independently measured speedup.

This block records the historical paired-test configuration and has deliberately not been relabelled. The current local launchers use `--spec-draft-n-max 10`; changing the label on the saved MTP7 measurements would make the comparison irreproducible.

## Dataset and scoring methodology

### MMLU-Pro

- Source: [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro).
- Full test split: 12,032 questions.
- Evaluated subset: fixed category-stratified sample of 100 questions.
- Prompting: five same-category validation examples with chain-of-thought.
- Generation: up to 1,024 reasoning tokens, followed by a second deterministic call requesting only `Answer: X`.
- Scoring: exact final option letter.

The two-stage final-letter extraction requires disclosure because it is not a standard public leaderboard protocol. It was introduced symmetrically for both variants after a one-stage pilot frequently reached the output cap without a parseable final letter. Pilot files are retained for audit but excluded from every result below.

### GSM8K

- Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), `main` configuration.
- Evaluated subset: fixed random sample of 100 out of 1,319 test questions.
- Prompting: eight fixed training demonstrations with step-by-step solutions.
- Scoring: normalized numeric value following the final `####` marker, with a conservative fallback parser.

### HumanEval

- Source and scorer: [`openai/human-eval`](https://github.com/openai/human-eval), commit `6d43fb980f9fee3c892a914eda09951f772ad10d`.
- Evaluated subset: fixed random sample of 40 problems.
- Metric: pass@1.
- Generated Python was executed against the official tests in `python:3.12-slim` with no network, a read-only root filesystem, dropped capabilities, no-new-privileges, one CPU, 256 MiB memory, a PID limit, and an eight-second timeout.

### IFEval

- Source and scorer: [`google-research/instruction_following_eval`](https://github.com/google-research/google-research/tree/master/instruction_following_eval), repository commit `1eb8bb0cbe5fd9072311ae3fd760e3644fee690b`.
- Evaluated subset: fixed random sample of 50 out of 541 prompts.
- Metrics: official strict/loose prompt-level and instruction-level scores.

### Long context

Nine synthetic exact-match cases combined three target lengths with three needle positions and output forms:

| Target | Needle position | Required output |
|---:|---|---|
| ~8K | Start | Exact secret string |
| ~8K | Middle | Strict JSON containing `secret` and `owner` |
| ~8K | End | Valid Python defining `get_secret()` |
| ~32K | Start | Exact secret string |
| ~32K | Middle | Strict JSON containing `secret` and `owner` |
| ~32K | End | Valid Python defining `get_secret()` |
| ~60K | Start | Exact secret string |
| ~60K | Middle | Strict JSON containing `secret` and `owner` |
| ~60K | End | Valid Python defining `get_secret()` |

The filler records explicitly stated that archive contents were data rather than instructions. Each prompt had a stored SHA256 digest.

### Tool calling

Every request exposed the same four OpenAI-compatible function schemas and used `tool_choice: auto`:

| Function | Required arguments | Cases |
|---|---|---:|
| `get_weather` | `city: string`, `unit: celsius\|fahrenheit` | 5 |
| `calculator` | `expression: string` | 5 |
| `lookup_order` | `order_id: string` | 5 |
| `search_docs` | `query: string`, `limit: integer` | 5 |

A case passed only when the first returned tool call had the exact expected function name and the parsed JSON arguments matched the expected object, including value types. Random tool-call IDs were intentionally ignored.

This suite tests simple single-step routing and argument extraction. It does not test execution of the function, multi-turn result handling, parallel calls, nested schemas, error recovery, ambiguous requests, or cases where the correct behavior is not to call a tool.

### Deterministic stability check

The same request was sent 100 times:

```text
Return exactly this text and nothing else: BENCHMARK_OK_42
```

The expected output, output hash, uniqueness, HTTP success, and runtime timings were recorded for every request.

## Main quality results

Wilson 95% confidence intervals are included for the sampled quality suites. Exact McNemar tests use paired per-item correctness between Q4 and Q6.

| Suite | N | Q4 | Q4 Wilson 95% CI | Q6 | Q6 Wilson 95% CI | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 100 | **76.0%** | 66.8–83.3% | 73.0% | 63.6–80.7% | 0.4531 |
| GSM8K | 100 | 93.0% | 86.3–96.6% | 93.0% | 86.3–96.6% | 1.0000 |
| HumanEval pass@1 | 40 | **95.0%** | 83.5–98.6% | 90.0% | 76.9–96.0% | 0.5000 |
| IFEval strict prompt | 50 | 82.0% | 69.2–90.2% | **84.0%** | 71.5–91.7% | 1.0000 |
| Long-context exact | 9 | 9/9 | — | 9/9 | — | — |
| Tool-call exact | 20 | 20/20 | — | 20/20 | — | — |
| Repeated prompt exact | 100 | 100/100 | — | 100/100 | — | — |

The confidence intervals overlap substantially, and no paired test approaches conventional significance. The defensible interpretation is **no demonstrated quality difference**, not that the variants are mathematically equivalent.

## Paired correctness

| Suite | Both pass | Q4 only | Q6 only | Both fail |
|---|---:|---:|---:|---:|
| MMLU-Pro | 71 | 5 | 2 | 22 |
| GSM8K | 92 | 1 | 1 | 6 |
| HumanEval | 36 | 2 | 0 | 2 |
| IFEval strict prompt | 39 | 2 | 3 | 6 |

Q4 gained five MMLU-Pro items that Q6 missed, while Q6 gained two that Q4 missed. On IFEval the direction reversed by one item. These discordant counts are too small for strong claims.

## Throughput and MTP acceptance

`Q6 speed delta` is calculated relative to Q4, so negative values mean Q6 was slower. Decode speed is token-weighted across each suite, not an unweighted average of per-request rates.

| Suite | Q4 decode tok/s | Q6 decode tok/s | Q6 speed delta | Q4 prefill tok/s | Q6 prefill tok/s | Q4 MTP accepted | Q6 MTP accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 21.77 | 17.72 | -18.6% | 653.28 | 608.42 | 46.94% | 47.21% |
| GSM8K | 27.79 | 22.51 | -19.0% | 683.64 | 650.03 | 64.42% | 64.39% |
| HumanEval | 27.69 | 22.98 | -17.0% | 361.58 | 288.88 | 64.25% | 66.72% |
| IFEval | 14.52 | 11.63 | -19.9% | 175.85 | 140.32 | 26.17% | 25.76% |
| Long context | 29.37 | 24.36 | -17.1% | 626.89 | 581.71 | 96.70% | 95.05% |
| Tool calling | 38.25 | 30.80 | -19.5% | 449.57 | 390.05 | 97.36% | 97.36% |
| Stability 100 | 35.59 | 28.68 | -19.4% | 82.63 | 75.32 | 100.00% | 100.00% |

### Complete workload totals

| Metric | Q4 | Q6 |
|---|---:|---:|
| Scored cases | 419 | 419 |
| API calls | 519 | 519 |
| Prompt tokens | 855,409 | 855,413 |
| Generated tokens | 112,331 | 112,952 |
| Weighted decode speed | **21.48 tok/s** | 17.43 tok/s |
| Draft tokens proposed | 187,048 | 187,730 |
| Draft tokens accepted | 86,379 | 86,900 |
| Overall MTP acceptance | 46.18% | 46.29% |

Q4's weighted decode throughput was 23.2% higher than Q6's. The nearly identical aggregate MTP acceptance suggests that the speed difference mainly follows the target quantization rather than better speculative acceptance for one variant.

The overall 46% acceptance does not contradict the 97–100% rates in structured suites. Open-ended reasoning and instruction-following produce less predictable token sequences, while exact short JSON/tool syntax and repeated fixed output are highly predictable.

## MMLU-Pro category breakdown

The per-category samples are only seven or eight questions each and should not be interpreted as stable category rankings.

| Category | N | Q4 correct | Q6 correct |
|---|---:|---:|---:|
| Biology | 8 | 6 | 6 |
| Business | 8 | 6 | 6 |
| Chemistry | 7 | 5 | 6 |
| Computer science | 7 | 5 | 5 |
| Economics | 7 | 7 | 7 |
| Engineering | 7 | 4 | 3 |
| Health | 7 | 7 | 6 |
| History | 7 | 5 | 5 |
| Law | 7 | 4 | 3 |
| Math | 7 | 5 | 5 |
| Other | 7 | 5 | 5 |
| Philosophy | 7 | 7 | 6 |
| Physics | 7 | 6 | 6 |
| Psychology | 7 | 4 | 4 |

## IFEval detail

| Metric | Q4 | Q6 |
|---|---:|---:|
| Strict prompt-level | 82.0% | **84.0%** |
| Strict instruction-level | 86.59% | **89.02%** |
| Loose prompt-level | 86.0% | **90.0%** |
| Loose instruction-level | 91.46% | **93.90%** |

IFEval is the only suite where Q6 was consistently ahead across all reported metrics, but the 50-prompt sample is insufficient to establish a reliable advantage.

## Long-context detail

All exact checks passed. Prefill throughput declined as context length increased, as expected.

| Target context | Cases | Q4 prefill tok/s | Q6 prefill tok/s | Q4 decode tok/s | Q6 decode tok/s |
|---:|---:|---:|---:|---:|---:|
| ~8K | 3 | 718.50 | 658.95 | 31.00 | 25.25 |
| ~32K | 3 | 660.64 | 613.32 | 30.68 | 25.45 |
| ~60K | 3 | 600.34 | 557.67 | 27.07 | 22.74 |

The often-quoted “about 29 tok/s” result is the Q4 aggregate for the predictable long-context suite. It should not be generalized to open-ended reasoning, where the same Q4 build averaged 21.77 tok/s on MMLU-Pro and 14.52 tok/s on IFEval.

## Tool-calling detail

Both variants returned 20/20 exact function names and argument objects.

| Function | Q4 passed | Q6 passed | Q4 decode tok/s | Q6 decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| `get_weather` | 5/5 | 5/5 | 39.36 | 31.57 | 100.00% |
| `calculator` | 5/5 | 5/5 | 35.93 | 29.20 | 93.88% |
| `lookup_order` | 5/5 | 5/5 | 39.88 | 31.97 | 97.28% |
| `search_docs` | 5/5 | 5/5 | 37.78 | 30.41 | 97.71% |
| **Total** | **20/20** | **20/20** | **38.25** | **30.80** | **97.36%** |

The tool result is encouraging but deliberately narrow. A future agentic evaluation should add multi-turn tool-result loops, parallel calls, nested schemas, failure recovery, similar function names, malicious tool output, and prompts for which no tool should be called.

## Short-run stability

For both variants:

- 100/100 requests returned the exact expected text.
- Only one unique output hash appeared.
- No malformed response was observed.
- MTP accepted 700/700 proposed draft tokens.

This is a deterministic consistency check, not a long-duration reliability claim. A 72-hour soak protocol is defined below.

## External MTP context: why “up to 3×” is workload-dependent

A separate [Qwen3.8 MTP walkthrough](https://www.youtube.com/watch?v=NjfHqiNHTxk) reports a best-case result of 167 tok/s on one agentic workload and 93 tok/s on ordinary text, an optimum near five draft tokens in that setup, and roughly 2–2.5 GB of additional VRAM use. Those figures belong to the video's software, hardware, prompt, and memory configuration; they are context rather than directly comparable measurements for this NVIDIA GB10 run.

The qualitative observations align with this experiment:

- Speculative candidates are verified by the target model and rejected when they do not match. The initial Q4/Q6 comparison did not include MTP-off, but the follow-up depth sweep did: all tested depths passed 9/9 objective checks and were semantically identical to baseline on 9/9 cases; byte identity was 8/9 or 9/9 depending on depth.
- Draft depth is a tunable rather than a universal constant. The original paired comparison fixed depth seven to isolate Q4 versus Q6; the follow-up tested depths 2–5 and 7–16 and found different optima for predictable and mixed workloads.
- Higher-temperature generation is harder to predict. This experiment deliberately used temperature zero for reproducibility and therefore should not be extrapolated to creative high-temperature sampling.
- Memory residency matters. The follow-up measured process RSS rather than driver-reported VRAM because this GB10 stack does not expose a usable VRAM counter: MTP7 added about 478 MiB after a short warm-up and about 1,092 MiB at the maximum observed RSS during the 8K/32K/64K sweep. These are unified-memory process measurements and are not directly comparable to the video's 2–2.5 GB VRAM figure.
- Prefix processing can dominate long agent sessions. The initial benchmark disabled prompt caching and showed prefill falling from 718.50 to 600.34 tok/s as Q4 grew from approximately 8K to 60K. The follow-up cache A/B below directly measures the much larger end-to-end effect in a progressive agent session.
- CLI drift can create a false “MTP enabled” run if an obsolete option is ignored. This run used the current `--spec-type draft-mtp` and `--spec-draft-n-max 7` arguments for the pinned `llama.cpp` commit. Non-zero `draft_n` and `draft_n_accepted` counters in every raw suite confirm that speculative drafting was active.

The video's headline should therefore be read as an achievable best case, not a portable multiplier. The present measurements reinforce the same point: structured output reached 97–100% draft acceptance, while IFEval reached only about 26%, and observed decode throughput changed with the workload even though the model, quantization, MTP depth, and sampling parameters stayed fixed.

## Local MTP depth and prefix-cache follow-up (2026-08-20)

These follow-up runs use the same Q4 model file, pinned `llama.cpp` binary, temperature zero, seed 42, full GPU offload, and Q8_0 K/V cache. They answer the optimization questions raised by the external walkthrough; they do not replace the larger Q4-versus-Q6 quality comparison above.

### Predictable 8K/32K/64K depth sweep

The first sweep used three objective workloads—integer sequence, strict JSON, and Python source—at approximately 8K, 32K, and 64K prompt lengths. MTP-off is the baseline. Selection requires 9/9 objective passes and semantic equivalence with baseline.

| Mode | Decode tok/s | Speedup | Acceptance | Minimum case tok/s | Objective | Exact vs baseline | Semantic vs baseline | Max RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 9.17 | 1.00x | — | 8.52 | 9/9 | 9/9 | 9/9 | 2268.6 |
| mtp2 | 18.86 | 2.06x | 99.46% | 15.55 | 9/9 | 9/9 | 9/9 | 3352.7 |
| mtp3 | 23.04 | 2.51x | 98.75% | 18.69 | 9/9 | 9/9 | 9/9 | 3350.5 |
| mtp4 | 26.53 | 2.89x | 98.62% | 21.99 | 9/9 | 8/9 | 9/9 | 3352.3 |
| mtp5 | 29.21 | 3.18x | 98.06% | 23.80 | 9/9 | 8/9 | 9/9 | 3353.9 |
| mtp7 | 33.32 | 3.63x | 93.55% | 25.27 | 9/9 | 8/9 | 9/9 | 3361.0 |
| mtp8 | 34.03 | 3.71x | 92.54% | 23.51 | 9/9 | 9/9 | 9/9 | 3402.3 |
| mtp9 | 35.51 | 3.87x | 90.44% | 25.22 | 9/9 | 9/9 | 9/9 | 3367.2 |
| mtp10 | 37.01 | 4.03x | 87.98% | 25.62 | 9/9 | 9/9 | 9/9 | 3401.9 |
| mtp11 | 37.66 | 4.11x | 85.23% | 25.30 | 9/9 | 9/9 | 9/9 | 3407.5 |
| mtp12 | 37.41 | 4.08x | 81.98% | 23.71 | 9/9 | 9/9 | 9/9 | 3372.7 |
| mtp13 | 38.05 | 4.15x | 80.72% | 22.45 | 9/9 | 9/9 | 9/9 | 3411.3 |
| mtp14 | 37.78 | 4.12x | 76.93% | 21.67 | 9/9 | 9/9 | 9/9 | 3372.5 |
| mtp15 | 37.94 | 4.14x | 74.74% | 21.11 | 9/9 | 9/9 | 9/9 | 3377.9 |
| mtp16 | 38.14 | 4.16x | 73.68% | 20.40 | 9/9 | 8/9 | 9/9 | 3376.3 |

By aggregate decode rate, MTP16 is the formal winner at 38.14 tok/s, 4.16x baseline. That headline is incomplete: it is only 0.24% faster than MTP13, its minimum case falls to 20.40 tok/s, acceptance falls to 73.68%, and byte identity is 8/9. MTP11 has a better mixed-case floor of 25.30 tok/s, while MTP16 is best treated as a specialized setting for long, regular generation.

### Mixed quality mini-sweep

The second paired run used frozen 10-case subsets of IFEval and GSM8K, 10 tool calls, and 10 ordinary chat prompts per mode (40 requests per mode, 240 total). Prefix caching was disabled. IFEval used the official strict scorer.

| Mode | IFEval strict | GSM8K | Tools | Valid chat | Decode tok/s | Speedup vs baseline | Acceptance | Warm RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 90% | 90% | 100% | 100% | 9.40 | 1.00x | — | 1704.7 |
| mtp5 | 90% | 90% | 100% | 100% | 19.64 | 2.09x | 47.32% | 2179.7 |
| mtp7 | 90% | 90% | 100% | 100% | 18.28 | 1.94x | 37.15% | 2182.7 |
| mtp11 | 90% | 90% | 100% | 100% | 15.38 | 1.64x | 24.29% | 2189.0 |
| mtp13 | 90% | 90% | 100% | 100% | 14.32 | 1.52x | 21.13% | 2192.9 |
| mtp16 | 100% | 90% | 100% | 100% | 12.67 | 1.35x | 16.72% | 2195.6 |

MTP5 is the mixed-workload throughput winner at 19.64 tok/s (2.09x baseline). MTP7 reaches 18.28 tok/s; depths 11, 13, and 16 become progressively slower as acceptance falls. MTP16 scored 10/10 rather than 9/10 on this small IFEval subset, but one prompt is far too little evidence for a quality claim; GSM8K, tools, and chat are identical across all six modes.

The memory result is also workload-dependent. Warm process RSS rose from 1,704.7 MiB at baseline to 2,182.7 MiB at MTP7 (+478.0 MiB). During the long-context sweep, maximum RSS rose from 2,268.6 to 3,361.0 MiB (+1,092.4 MiB). NVIDIA GB10 uses unified memory here and `nvidia-smi` does not expose a comparable dedicated-VRAM allocation, so these numbers must be reported as process RSS.

### Prefix-cache A/B for a progressive agent session

The cache test used MTP7, a tokenizer-measured 18,243-token static system prefix, 12 progressive tool-calling turns, and 24 requests per mode. Each mode started from a fresh server.

| Mode | Task quality | Prompt tokens evaluated | Prompt seconds | Decode tok/s | End-to-end seconds |
|---|---:|---:|---:|---:|---:|
| cache off | 12/12 | 460436 | 703.17 | 34.86 | 734.84 |
| cache on | 12/12 | 39381 | 72.11 | 35.12 | 103.10 |

`cache_prompt=true` reduced prompt-processing time by 89.74% and improved end-to-end wall time by 7.13x, with identical 12/12 quality and the same MTP acceptance. For this long-prefix agent scenario, prefix caching matters much more than increasing MTP depth beyond seven.

### Operational decision

- The predeclared soak retained MTP7 so its protocol was not changed after observing optimization results. It was later stopped by the operator after 6 h 17 min; its partial measurements remain labelled MTP7.
- Use MTP10 as the new local Q4 and Q6 launcher default. It provides a stronger predictable-workload average than MTP7 (37.01 vs 33.32 tok/s), a slightly better measured minimum-case rate (25.62 vs 25.27 tok/s), 87.98% acceptance, and 9/9 objective and semantic checks.
- MTP5 remains the measured winner of the small mixed chat/tool mini-sweep. MTP10 was not included in that mini-sweep, so adopting it is an operational balance for this host rather than a claim that it is universally optimal.
- Reserve MTP13–16 for explicitly regular, long-generation workloads; do not use their best-case average as a universal speed claim.
- Enable prefix caching for stable long prefixes and progressive agent histories, while monitoring cache residency and correctness.

## Invalid pilot handling

An early Q4 MMLU-Pro pilot used a single generation stage. Some answers reached the output limit without a trustworthy final choice. Those checkpoints were invalidated rather than silently scored, and the final paired run used the same two-stage procedure for both Q4 and Q6.

The excluded diagnostic files remain in the private local audit directory and are intentionally not included in this public repository:

- `q4_one_stage_invalid.json`
- `q4_smoke_one_stage.json`
- `q4_smoke_before_adapter_fix.json`

They are not read by `make_report.py` and do not contribute to any table in this article.

## Interpretation

The most defensible conclusions from this run are:

1. Q6 did not demonstrate a statistically reliable quality advantage over Q4 on these fixed subsets.
2. Q6 was consistently slower, producing 17.0–19.9% fewer decode tokens per second in every suite.
3. Q4 is the practical default for this NVIDIA GB10 deployment.
4. Q6 remains worth testing on a larger IFEval sample or a task distribution known to reward its slightly higher instruction-following scores.
5. MTP acceptance is a workload characteristic. A single acceptance number should not be used to predict speed across unrelated tasks.

## Limitations

- MMLU-Pro, GSM8K, HumanEval, and IFEval used fixed subsets rather than their full evaluation sets.
- Only one host and one `llama.cpp` commit were measured.
- The experiment used deterministic temperature-zero decoding only.
- Server parallelism was one; throughput under concurrent production traffic was not evaluated.
- No MTP-off baseline was included in this run.
- The long-context suite used synthetic filler and exact retrieval rather than document QA over natural corpora.
- The tool suite tested selection and serialization but did not execute tools or continue the conversation with tool results.
- One pass per quantization cannot quantify run-to-run system variance.
- Quality differences are underpowered at these sample sizes.

## Reproducing the run

The repository layout expected by the scripts is:

```text
data/
results/
vendor/
fetch_data.py
run_benchmark.py
make_report.py
```

Update the `SERVER` and `MODEL_FILES` constants in `run_benchmark.py` for the local installation, then run from a clean result directory:

```bash
python3 fetch_data.py
python3 run_benchmark.py --quant q4 --suite all
python3 run_benchmark.py --quant q6 --suite all
python3 make_report.py
```

The runner checkpoints each completed case, allowing interrupted runs to resume. Use a fresh checkout or archive prior result JSON before starting an independent replication.

### Result integrity

```text
q4.json      06d328340309732e66be2d77c2c15df8e8f164c96a5be07aabdb0e883cdfd12a
q6.json      b20917c30186f77a513e69f368a1ba2dfa842d4d70de94724178f68536ab472e
summary.json 76081f2001c378f954317f41f1bdc39184d0bb4020ab007cf04e3f0574436523
```

Relevant artifacts:

- [`run_benchmark.py`](run_benchmark.py)
- [`fetch_data.py`](fetch_data.py)
- [`make_report.py`](make_report.py)
- [`data/manifest.json`](data/manifest.json)
- [`results/q4.json`](results/q4.json)
- [`results/q6.json`](results/q6.json)
- [`results/summary.json`](results/summary.json)

## Harness v1

The one-off scripts above are hard-wired to one machine. `sparkbench/` is the reusable replacement: the same suites and scorers, driven by a TOML config against any OpenAI-compatible `/v1/chat/completions` server (llama.cpp `llama-server`, SGLang, vLLM), writing one uniform result format. It needs Python 3.12 and the standard library only (pytest, ruff and mypy for development).

```text
sparkbench/            harness package (python -m sparkbench ...)
  suites/              the 7 suites and scorers moved from run_benchmark.py, logic unchanged
  importers/           converters for every historical result file
schemas/result.v1.json JSON Schema of sparkbench.result.v1
examples/*.toml        run and sweep configs: all host paths, model files and server flags live here
results/v1/            historical results converted to v1 (28 files; sources untouched)
tests/                 pytest suite: fake OpenAI server, scorer fixtures, README regression
```

### Commands

```bash
python -m sparkbench run --config examples/llamacpp-gb10-q4-mtp7.toml             # all 7 suites
python -m sparkbench run --config examples/llamacpp-gb10-q4-mtp7.toml --smoke     # 2-3 cases per suite
python -m sparkbench run --config X.toml --suite tools --suite stability_100      # selected suites
python -m sparkbench sweep --config examples/sweep-qwen38-sglang.toml             # parameter grid, see "Sweep"
python -m sparkbench report A.json B.json [--out report.md] [--json summary.json] # paired A/B report
python -m sparkbench compare-hosts A.json B.json [--out hosts.md] [--fail-on-diff]
python -m sparkbench validate results/v1/*.json
python -m sparkbench import                                                       # rebuild results/v1/
```

`run` checkpoints after every case to `<output>.raw.json` (full server responses) and rewrites `<output>.json` (v1). Re-running the same command resumes; a checkpoint from a different config is refused unless `--fresh` is given.

### Config: llama.cpp

[`examples/llamacpp-gb10-q4-mtp7.toml`](examples/llamacpp-gb10-q4-mtp7.toml) is the published Q4 run expressed as config; its `config_sha256` equals the one recorded in `results/v1/main_q4_mtp7.json`. Abridged:

```toml
[run]
label = "q4-mtp7"
output = "results/runs/q4-mtp7.json"

[model]
name = "qwen3.8-MTP:27b"            # model id sent in requests
file = "/usr/share/ollama/.ollama/models/blobs/sha256-bee238bb..."
sha256 = "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"

[runtime]
name = "llama.cpp"
version = "d2f83055d6e3b379b5d34c4837122a918cf402c2"

[backend]
base_url = "http://127.0.0.1:18084"
tokenizer = "llama.cpp"             # /tokenize, used to size long-context prompts
extra_body = { cache_prompt = false }

[server]                            # optional: omit to use an already running server
command = ["/home/admin/llama.cpp-main-20260810/build/bin/llama-server",
           "--model", "...", "--alias", "qwen3.8-MTP:27b", "--port", "18084",
           "--ctx-size", "65536", "--spec-type", "draft-mtp", "--spec-draft-n-max", "7", "..."]

[request]
temperature = 0
seed = 42

[parameters]                        # recorded verbatim in the result
mtp_depth = 7
context = 65536
```

### Config: SGLang

[`examples/sglang-q4.toml`](examples/sglang-q4.toml) (and [`examples/vllm-q4.toml`](examples/vllm-q4.toml)) use the same suites. The paths and speculative-decoding flags there are placeholders to adapt to the installed version:

```toml
[model]
name = "qwen3.8-27b"                 # --served-model-name

[runtime]
name = "sglang"

[backend]
base_url = "http://127.0.0.1:30000"
tokenizer = "openai-tokenize"        # POST /tokenize {"model", "prompt"}
extra_body = {}                      # no llama.cpp-only request fields

[server]
command = ["python3", "-m", "sglang.launch_server", "--model-path", "/models/Qwen3.8-27B-AWQ",
           "--served-model-name", "qwen3.8-27b", "--port", "30000", "--context-length", "65536",
           "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "7"]
ready_timeout = 1200
```

SGLang and vLLM return no llama.cpp `timings`, so decode tok/s is `completion_tokens / request wall time` (marked `timing_source: "wall"`; it includes prefill) and MTP acceptance is not available. Compare such runs with runs of the same runtime.

### Comparing two hosts

Run the same config on both machines. Only host-specific values (paths, URL, `[host]`) may differ: `workload_sha256` ignores them, so `compare-hosts` can confirm that both sides measured the same workload.

```bash
# on spark-1
python -m sparkbench run --config examples/llamacpp-gb10-q4-mtp7.toml --output results/runs/spark1-q4.json
# on spark-2 (same file; edit [model] file / [server] command only if paths differ)
python -m sparkbench run --config examples/llamacpp-gb10-q4-mtp7.toml --output results/runs/spark2-q4.json
# on either machine, after copying both JSON files
python -m sparkbench validate results/runs/spark1-q4.json results/runs/spark2-q4.json
python -m sparkbench compare-hosts results/runs/spark1-q4.json results/runs/spark2-q4.json --out results/runs/spark1-vs-spark2.md
```

The comparison reports decode and prefill tok/s per suite with the B/A delta and the median per-case ratio, then checks answer identity case by case: identical outputs, differences that change a number in the answer (listed separately with the numbers only on one side and the final number on each side), text-only differences, and pass/fail flips. `--fail-on-diff` returns exit code 1 if any paired answer differs.

### Sweep

`sweep` runs one config over a grid of server and request parameters (speculative algorithm, draft block size, `--mem-fraction-static`, `--max-running-requests`, context length, thinking on/off, ...). For every grid point it substitutes the values into the base config, starts the server from `[server] command`, waits for `/health` up to `ready_timeout`, runs the suites through the normal `run` path and stops the server (the whole process group, so worker processes do not keep the port). [`examples/sweep-qwen38-sglang.toml`](examples/sweep-qwen38-sglang.toml) sweeps SGLang with no speculation, MTP (NEXTN), EAGLE3, DFLASH and DSPARK, two draft block sizes and thinking on/off: 18 points.

```bash
python -m sparkbench sweep --config examples/sweep-qwen38-sglang.toml --dry-run      # points, commands, what would run
python -m sparkbench sweep --config examples/sweep-qwen38-sglang.toml --smoke        # 2-3 cases per suite
python -m sparkbench sweep --config examples/sweep-qwen38-sglang.toml                # run or resume
python -m sparkbench sweep --config examples/sweep-qwen38-sglang.toml --report-only  # rebuild the report
```

A sweep file has a base run config (inline `[base.*]` tables, or `[sweep] base_config = "file.toml"` relative to the sweep file), a grid, and a selection rule:

```toml
[sweep]
name = "qwen38-sglang"
output_dir = "results/sweeps/qwen38-sglang"
suites = ["gsm8k", "ifeval", "tools", "long_context"]   # overrides [run] suites
ready_timeout = 1500                                    # per point, seconds

[grid]
spec = [
  { label = "off", spec_args = [] },
  { label = "mtp", spec_args = ["--speculative-algorithm", "NEXTN", "--speculative-num-draft-tokens", "{block}"] },
]
draft = [{ label = "b4", block = 4, steps = 3 }, { label = "b8", block = 8, steps = 7 }]
mem_fraction = [0.85]
thinking = [false, true]

[[exclude]]                  # skip combinations; a list value means "any of"
spec = "off"
draft = "b8"

[select]
rule = "max_speed_within_baseline"
baseline = { spec = "off", thinking = false }
max_quality_drop_pp = 1.0

[base.server]
command = ["python3", "-m", "sglang.launch_server", "--mem-fraction-static", "{mem_fraction}", "{spec_args}"]

[base.backend]
base_url = "http://127.0.0.1:30000"
extra_body = { chat_template_kwargs = { enable_thinking = "{thinking}" } }
```

Substitution rules:

- Each grid axis defines a placeholder `{axis}`. A grid value can be a scalar or a table with a `label`; the label names the point and the table's other keys become placeholders too (`{spec_args}`, `{block}` above).
- A string that is exactly `"{name}"` keeps the value's TOML type: `enable_thinking = "{thinking}"` becomes a boolean, and a list is spliced into the enclosing list (`"{spec_args}"` in the command). Inside a longer string, the value is formatted as text (booleans as `true`/`false`). `[server] command` items are always converted to strings.
- Unknown placeholders are an error. Write `{{` and `}}` for literal braces. `{point}` is the point id.
- `[[exclude]]` removes grid combinations, and `[[include]]` adds points outside the grid. An include gives every axis; a table-valued axis takes an existing label or a full table.

Output in `output_dir`:

- `points/<id>.json`: `sparkbench.result.v1` of the point. The id is `axis-value__axis-value...`. The grid values are recorded in `parameters.sweep_point` and so are part of `workload_sha256`.
- `points/<id>.server.log`: server stdout/stderr.
- `points/<id>.failed.json` (`sparkbench.sweep_failure.v1`): written when the server exits, never answers `/health` in time, or the run fails. It holds the stage (`server_start`, `run` or `invalid_result`), the error and traceback, the command and the last 4 KB of the server log. The sweep moves on to the next point, and the exit code is 1 when any point failed.
- `report.md` and `report.json` (`sparkbench.sweep_report.v1`): the report.

Resume: re-running the command skips a point when a result exists that validates against the schema, is complete, covers the requested suites and has the same `workload_sha256` and smoke mode. Failed and missing points run again, and an interrupted point resumes from its `.raw.json` checkpoint. Any change to grid values, `[parameters]`, the model, the request settings or `extra_body` changes the hash and re-runs the point. `workload_sha256` does not cover the server command, because it holds host paths. After editing a flag that no grid axis controls, use `--fresh` or record the flag in `[parameters]`. Before each start, the sweep checks that nothing already answers the health URL, so a previous server that is still shutting down is never benchmarked by mistake.

The report has a table of points: grid values, status, quality, per-suite quality, overall weighted decode tok/s, draft acceptance, and delta against the reference point. It also lists the Pareto front over quality × tok/s (points that no other point beats on both), the recommended point with the rule written out, and the failed points with their last log line. Quality is the mean of the suite `quality` scores (`quality = "micro"` gives passed/scored cases over all suites). Rules:

| `rule` | Chooses |
|---|---|
| `max_speed_within_baseline` (default) | highest tok/s among points whose quality is at most `max_quality_drop_pp` below the `baseline` point (a table of axis values that matches exactly one point, or `"best"` for the best-quality point) |
| `max_speed_min_quality` | highest tok/s among points with quality >= `min_quality_pct` |
| `max_quality` | highest quality, ties broken by tok/s |

Only measured points are ranked. If the baseline point failed, the report gives no recommendation and says why. With SGLang/vLLM, tok/s is wall-clock based (see above). The sweep compares points on one runtime, so this is consistent within a sweep.

### Result format `sparkbench.result.v1`

One JSON file per run, validated by [`schemas/result.v1.json`](schemas/result.v1.json). It keeps the `results/q4.json` layout (`schema`, `quant`, `model`, `runtime`, `parameters`, `suites`, timings) and adds `host` (name, GPU, driver, arch, kernel), `backend` (URL, request extras, tokenizer), `config_sha256`, `workload_sha256`, the parsed config, and `source` (run or import, with source file hashes). Every suite has `quality`, `verdicts`, `metrics`, an `aggregate` computed exactly as before, and uniform per-case records: `id`, `verdict`, `prompt_sha256`, prompt/generated/draft tokens, wall/prefill/decode time, tok/s, finish reason, content and `content_sha256`, plus tool calls and earlier turns when present.

`results/v1/` holds the historical runs converted by `python -m sparkbench import`: the main Q4/Q6 runs, 15 MTP sweep modes, the Q6 comparison run, 6 mixed mini-sweep modes, the prefix-cache A/B, and the partial soak (one file per quant, with per-request records from `requests.ndjson`). `python -m sparkbench report results/v1/main_q4_mtp7.json results/v1/main_q6_mtp7.json` reproduces the tables in this README.

### Data and scorer dependencies

The suites read the same inputs as before: `data/` from `fetch_data.py`, HumanEval from `vendor/human-eval`, and Google's IFEval scorer from `vendor/google-research` (it needs `nltk`, `langdetect` and `absl-py` in the environment that runs IFEval). HumanEval completions run in the same networkless read-only Docker sandbox (`python:3.12-slim`). Paths can be changed in `[data]`.

### Development

```bash
python3.12 -m venv .venv && .venv/bin/pip install pytest ruff mypy
.venv/bin/python -m pytest          # no network or GPU needed
.venv/bin/ruff check .
.venv/bin/mypy --strict sparkbench
```

## 72-hour soak test (stopped early; partial results)

The following protocol was defined before the run. The validated service started at **2026-08-20 07:44:53 UTC**; the first Q4 preflight passed with exact output, `draft_n=357`, `draft_n_accepted=353`, and the MTP fail-closed guard active. The operator stopped the service at approximately **2026-08-20 14:02:29 UTC**, after about 6 h 17 min. This does not constitute a completed 72-hour reliability result. Raw partial data are retained and reported without changing the predeclared criteria.

An earlier engineering preflight started at 05:41:59 UTC and was invalidated before the 72-hour clock: it exposed an overlong reasoning-format contract, an estimated “60K” prompt that actually tokenized to 111,047 tokens, and a graceful-stop timeout. Its artifacts are preserved in the private local audit directory and excluded from both the public repository and all soak totals. Before restart, all five reasoning contracts passed on both Q4 and Q6, corrected long-context prompts tokenized to 7,985/31,986/59,886 tokens and passed on both models, and a 96-request stop-smoke confirmed clean signal handling.

### Goal

Test operational stability, not benchmark quality: crashes, hangs, memory growth, malformed responses, deterministic drift, throughput degradation, MTP behavior, and GPU/runtime errors during sustained mixed inference.

### Schedule

- Total elapsed test time: 72 hours.
- Q4 exposure: 36 measured hours.
- Q6 exposure: 36 measured hours.
- Block duration: six hours.
- One model resident at a time.
- Counterbalanced block order: `Q4, Q6, Q6, Q4, Q6, Q4, Q4, Q6, Q4, Q6, Q6, Q4`.
- Before measured time begins, a preflight probe will require non-zero `draft_n` and `draft_n_accepted` counters, record the exact server command line, and fail closed if MTP telemetry is absent.
- A short MTP-off/MTP-on deterministic probe will compare output hashes before the soak. It will be reported separately and will not be mixed into the 72 measured hours.
- A fixed warm-up interval after every model load will be recorded but excluded from throughput statistics.
- The same request corpus, order seed, server flags, and monitoring cadence will be used for both variants.

### Proposed workload mix

| Workload | Share | Purpose |
|---|---:|---|
| Short deterministic prompts | 25% | Detect output drift and malformed responses |
| Reasoning and mathematics | 20% | Exercise lower-acceptance open generation |
| Code generation | 15% | Exercise syntax-sensitive output |
| Tool calls, including negative cases | 20% | Validate routing and JSON under repetition |
| ~8K context | 10% | Sustained moderate prefill |
| ~32K context | 7% | KV-cache and memory pressure |
| ~60K context | 3% | Near-limit long-context pressure |

The corpus will be finite, versioned, and repeated in a fixed seeded order. Tool coverage will be expanded beyond the current simple suite to include no-call decisions, multi-turn tool results, invalid tool results, nested arguments, similar function names, and controlled prompt-injection strings inside tool output.

### Per-request telemetry

- UTC timestamp and block identifier.
- Quantization, model hash, runtime commit, and server PID.
- HTTP status, timeout category, retry count, and wall latency.
- Prompt and generated tokens.
- Prefill tok/s and decode tok/s.
- Draft tokens proposed and accepted.
- MTP-active guard result and the exact speculative-decoding CLI arguments.
- Exact-output or task-specific correctness result.
- Output SHA256 for deterministic probes.
- Process RSS and available system memory.
- GPU utilization, temperature, clocks, and power where the driver exposes them.
- Server restart count and model reload time.
- Kernel/NVIDIA Xid, OOM, and runtime error events.

### Predeclared acceptance criteria

| Criterion | Pass condition |
|---|---|
| Server crashes/restarts | 0 unplanned |
| OOM or NVIDIA Xid events | 0 |
| Missing MTP telemetry or zero drafting during the preflight | 0 accepted blocks; the affected block must not start |
| Corrupted/malformed structured outputs on deterministic probes | 0 |
| Deterministic probe accuracy | 100% |
| Request success rate | At least 99.9%, with every failure classified |
| Memory growth after warm-up | No sustained monotonic growth; end-of-block RSS within 5% of its post-warm-up baseline |
| Median task-normalized decode degradation | No more than 5% from the first comparable block |
| p95 task-normalized latency degradation | No more than 10% from the first comparable block |
| MTP acceptance shift | No unexplained sustained change greater than 5 percentage points for the same workload class |

Any failure will be retained in raw logs; retries will not erase the original event.

### Partial results at operator stop

| Metric | Q4 | Q6 |
|---|---:|---:|
| Observed request window | 5 h 59 min | 16 min 36 s |
| Completed requests | 2,372 | 101 |
| Passed requests | 2,372 | 101 |
| Success rate | 100% | 100% |
| Timeouts or response errors | 0 | 0 |
| Deterministic mismatches | 0 | 0 |
| Generated tokens | 77,962 | 3,330 |
| Token-weighted decode rate | 29.80 tok/s | 24.13 tok/s |
| Overall MTP acceptance | 71.09% | 71.11% |
| Peak process VmHWM | 17,445.9 MiB | 25,076.4 MiB |

Only the first six-hour Q4 block completed. Q6 ran for roughly 17 minutes, so the two rows are not balanced and must not be used as a completed Q4/Q6 reliability comparison. The load generator, fixed workload manifest, raw per-request NDJSON, periodic system telemetry, server logs, and generated summary are retained with the partial run.

## Conclusion

On this controlled NVIDIA GB10 run, Q4 preserved the measured quality of Q6 while delivering materially higher throughput and using a substantially smaller model file. The small Q6 IFEval advantage is interesting but not statistically established. Q4 is therefore the default quantization for this deployment, and the local launcher now uses MTP10 as the measured operational compromise. The stopped soak supplies useful partial telemetry but no 72-hour reliability claim.
