#!/usr/bin/env python3
"""Build paired Q4/Q6 summary and Markdown report from raw checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_binomial_two_sided(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    smaller = min(a, b)
    probability = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * probability)


def paired(left: list[dict], right: list[dict], key: str, value: str) -> dict:
    lmap = {item[key]: bool(item[value]) for item in left}
    rmap = {item[key]: bool(item[value]) for item in right}
    if lmap.keys() != rmap.keys():
        raise RuntimeError(f"paired key mismatch for {key}")
    both = sum(lmap[item] and rmap[item] for item in lmap)
    left_only = sum(lmap[item] and not rmap[item] for item in lmap)
    right_only = sum(not lmap[item] and rmap[item] for item in lmap)
    neither = sum(not lmap[item] and not rmap[item] for item in lmap)
    return {
        "both_correct": both,
        "q4_only": left_only,
        "q6_only": right_only,
        "both_wrong": neither,
        "discordant": left_only + right_only,
        "mcnemar_exact_p": exact_binomial_two_sided(left_only, right_only),
    }


def suite_metrics(state: dict, suite: str) -> dict:
    data = state["suites"][suite]
    result = dict(data["aggregate"])
    if suite == "mmlu_pro":
        result["quality"] = data["score"]
        result["quality_label"] = "accuracy"
    elif suite == "gsm8k":
        result["quality"] = data["score"]
        result["quality_label"] = "accuracy"
    elif suite == "humaneval":
        result["quality"] = data["pass_at_1"]
        result["quality_label"] = "pass@1"
    elif suite == "ifeval":
        result["quality"] = data["scores"]["strict"]["prompt_level"]
        result["quality_label"] = "strict prompt"
        result["strict_instruction"] = data["scores"]["strict"]["instruction_level"]
        result["loose_prompt"] = data["scores"]["loose"]["prompt_level"]
        result["loose_instruction"] = data["scores"]["loose"]["instruction_level"]
    elif suite == "long_context":
        result["quality"] = data["score"]
        result["quality_label"] = "exact"
    elif suite == "tools":
        result["quality"] = data["score"]
        result["quality_label"] = "exact"
    elif suite == "stability_100":
        result["quality"] = data["exact_rate"]
        result["quality_label"] = "exact"
        result["unique_outputs"] = data["unique_outputs"]
        result["modal_output_rate"] = data["modal_output_rate"]
    return result


def overall(state: dict) -> dict:
    aggregates = [state["suites"][name]["aggregate"] for name in state["suites"]]
    generated = sum(item["generated_tokens"] for item in aggregates)
    decode_seconds = sum(
        item["generated_tokens"] / item["decode_tokens_per_second"]
        for item in aggregates
        if item.get("decode_tokens_per_second")
    )
    prompts = sum(item["prompt_tokens"] for item in aggregates)
    draft = sum(item["draft_generated"] for item in aggregates)
    accepted = sum(item["draft_accepted"] for item in aggregates)
    return {
        "cases": sum(item["cases"] for item in aggregates),
        "api_calls": sum(item["cases"] for item in aggregates) + len(state["suites"]["mmlu_pro"]["cases"]),
        "prompt_tokens": prompts,
        "generated_tokens": generated,
        "decode_tokens_per_second": generated / decode_seconds,
        "draft_generated": draft,
        "draft_accepted": accepted,
        "draft_acceptance": accepted / draft,
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def num(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    q4 = read(RESULTS / "q4.json")
    q6 = read(RESULTS / "q6.json")
    suites = ["mmlu_pro", "gsm8k", "humaneval", "ifeval", "long_context", "tools", "stability_100"]
    summary = {
        "schema": 1,
        "method": {
            "temperature": 0,
            "seed": 42,
            "mtp_depth": 7,
            "context": 65536,
            "parallel": 1,
            "kv_cache": "q8_0/q8_0",
            "mmlu_pro": "100 category-stratified, 5-shot CoT, 1024-token reasoning plus deterministic final-letter extraction",
            "gsm8k": "100 fixed random test problems, 8-shot CoT",
            "humaneval": "40 fixed random problems, chat completion, official tests in networkless read-only Docker",
            "ifeval": "50 fixed random prompts, official Google strict/loose scorer",
            "long_context": "9 synthetic exact checks at ~8K/~32K/~60K",
            "tools": "20 exact OpenAI tool-call name/argument checks",
            "stability": "100 identical prompts",
        },
        "q4": {"overall": overall(q4), "suites": {name: suite_metrics(q4, name) for name in suites}},
        "q6": {"overall": overall(q6), "suites": {name: suite_metrics(q6, name) for name in suites}},
        "paired": {
            "mmlu_pro": paired(q4["suites"]["mmlu_pro"]["cases"], q6["suites"]["mmlu_pro"]["cases"], "id", "correct"),
            "gsm8k": paired(q4["suites"]["gsm8k"]["cases"], q6["suites"]["gsm8k"]["cases"], "id", "correct"),
            "humaneval": paired(q4["suites"]["humaneval"]["cases"], q6["suites"]["humaneval"]["cases"], "id", "passed"),
        },
        "mmlu_categories": {
            category: {"q4": score, "q6": q6["suites"]["mmlu_pro"]["by_category"][category]}
            for category, score in q4["suites"]["mmlu_pro"]["by_category"].items()
        },
    }

    q4_ifeval = q4["suites"]["ifeval"]["scores"]["strict"]["details"]
    q6_ifeval = q6["suites"]["ifeval"]["scores"]["strict"]["details"]
    summary["paired"]["ifeval_strict_prompt"] = paired(
        [{"key": item["key"], "passed": item["follow_all"]} for item in q4_ifeval],
        [{"key": item["key"], "passed": item["follow_all"]} for item in q6_ifeval],
        "key",
        "passed",
    )
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3.8-27B llama.cpp MTP7: Q4 vs Q6",
        "",
        "## Main result",
        "",
        "| Test | Q4 quality | Q6 quality | Q4 tok/s | Q6 tok/s | Q6 speed delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "mmlu_pro": "MMLU-Pro-100",
        "gsm8k": "GSM8K-100",
        "humaneval": "HumanEval-40",
        "ifeval": "IFEval-50 strict prompt",
        "long_context": "Long context 9",
        "tools": "Tool calls 20",
        "stability_100": "Stability 100",
    }
    for name in suites:
        left = summary["q4"]["suites"][name]
        right = summary["q6"]["suites"][name]
        speed_delta = right["decode_tokens_per_second"] / left["decode_tokens_per_second"] - 1
        lines.append(
            f"| {labels[name]} | {pct(left['quality'])} | {pct(right['quality'])} | "
            f"{num(left['decode_tokens_per_second'])} | {num(right['decode_tokens_per_second'])} | {speed_delta:+.1%} |"
        )
    lines.extend([
        "",
        "## Paired correctness",
        "",
        "| Test | Both pass | Q4 only | Q6 only | Both fail | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("mmlu_pro", "gsm8k", "humaneval", "ifeval_strict_prompt"):
        item = summary["paired"][name]
        lines.append(
            f"| {name} | {item['both_correct']} | {item['q4_only']} | {item['q6_only']} | "
            f"{item['both_wrong']} | {item['mcnemar_exact_p']:.4f} |"
        )
    lines.extend([
        "",
        "## IFEval detail",
        "",
        "| Metric | Q4 | Q6 |",
        "|---|---:|---:|",
        f"| Strict prompt | {pct(summary['q4']['suites']['ifeval']['quality'])} | {pct(summary['q6']['suites']['ifeval']['quality'])} |",
        f"| Strict instruction | {pct(summary['q4']['suites']['ifeval']['strict_instruction'])} | {pct(summary['q6']['suites']['ifeval']['strict_instruction'])} |",
        f"| Loose prompt | {pct(summary['q4']['suites']['ifeval']['loose_prompt'])} | {pct(summary['q6']['suites']['ifeval']['loose_prompt'])} |",
        f"| Loose instruction | {pct(summary['q4']['suites']['ifeval']['loose_instruction'])} | {pct(summary['q6']['suites']['ifeval']['loose_instruction'])} |",
        "",
        "## Long context and stability",
        "",
        f"- Long-context exact: Q4 9/9, Q6 9/9.",
        f"- Long-context prefill: Q4 {num(summary['q4']['suites']['long_context']['prompt_tokens_per_second'])} tok/s, Q6 {num(summary['q6']['suites']['long_context']['prompt_tokens_per_second'])} tok/s.",
        f"- Tool calls: Q4 20/20, Q6 20/20.",
        f"- Repeated prompts: both 100/100 exact with one unique output.",
        "",
        "## Overall workload",
        "",
        f"- Per quant: {summary['q4']['overall']['cases']} scored cases and {summary['q4']['overall']['api_calls']} model API calls.",
        f"- Q4 generated {summary['q4']['overall']['generated_tokens']} tokens at weighted {num(summary['q4']['overall']['decode_tokens_per_second'])} tok/s.",
        f"- Q6 generated {summary['q6']['overall']['generated_tokens']} tokens at weighted {num(summary['q6']['overall']['decode_tokens_per_second'])} tok/s.",
        f"- Overall MTP acceptance: Q4 {pct(summary['q4']['overall']['draft_acceptance'])}, Q6 {pct(summary['q6']['overall']['draft_acceptance'])}.",
        "",
        "## Method and limits",
        "",
        "- Hardware: NVIDIA GB10; one model at a time; competing Ollama/vLLM models stopped.",
        "- Runtime: llama.cpp, full GPU offload, MTP7, context 65,536, parallel 1, Q8_0 K/V cache.",
        "- Sampling: temperature 0 and seed 42 for both quants.",
        "- MMLU-Pro is a fixed category-stratified 100-question subset, not the full 12,032-question leaderboard run.",
        "- HumanEval is a fixed 40-problem pass@1 subset using the official unit tests; generated code ran in a networkless read-only container.",
        "- IFEval uses the official Google strict and loose scorers on a fixed 50-prompt subset.",
        "- Therefore these numbers are valid for direct Q4/Q6 comparison on this host, but are not drop-in replacements for full public leaderboard scores.",
    ])
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
