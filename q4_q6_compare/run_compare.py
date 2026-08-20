#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/admin/qwen38_q4_q6_compare_20260817")
SWEEP_ROOT = Path("/home/admin/qwen38_mtp_sweep_20260817")
Q4_RESULTS_PATH = SWEEP_ROOT / "results.json"
Q6_MODEL = Path(
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-739202186fd9389bb58497c58b56c8a0d4253d99d20131e6a0427e363e678fc8"
)
RESULTS_PATH = ROOT / "results.json"
REPORT_PATH = ROOT / "REPORT.md"
LOG_PATH = ROOT / "q6_mtp7_server.log"
PORT = 18083

sys.path.insert(0, str(SWEEP_ROOT))
import run_sweep as sweep  # noqa: E402


def save(state: dict) -> None:
    RESULTS_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def aggregate_by_context(cases: list[dict]) -> dict:
    return {
        context: sweep.aggregate_cases(
            [item for item in cases if item["context"] == context]
        )
        for context, _ in sweep.CONTEXT_TARGETS
    }


def write_report(state: dict, q4_mode: dict) -> None:
    q4_aggregate = q4_mode["aggregate"]
    q6_aggregate = state["q6_mtp7"]["aggregate"]
    q6_total = q6_aggregate["total"]
    speed_ratio = (
        q6_aggregate["decode_tokens_per_second"]
        / q4_aggregate["decode_tokens_per_second"]
    )
    prompt_ratio = (
        q6_aggregate["prompt_tokens_per_second"]
        / q4_aggregate["prompt_tokens_per_second"]
    )
    lines = [
        "# Qwen3.8-27B Q4 vs Q6 with MTP7",
        "",
        f"- Date: {state['completed_at_utc']}",
        "- Hardware: NVIDIA GB10",
        f"- Context allocation: {sweep.CTX_SIZE} tokens",
        "- KV cache: Q8_0 K/V; all layers on GPU; flash attention on",
        "- Sampling: temperature 0, seed 42",
        "- Workloads: sequence, strict JSON, Python code at ~8K/~32K/~64K",
        "- Every Q6 prompt hash was verified against the saved Q4 MTP7 run",
        "",
        "## Overall",
        "",
        "| Quant | Decode tok/s | Relative to Q4 | Prompt tok/s | Acceptance | Quality | Exact vs Q4 | Semantic vs Q4 | Min case tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Q4 UD-Q4_K_XL | {q4_aggregate['decode_tokens_per_second']:.2f} | "
            f"1.00x | {q4_aggregate['prompt_tokens_per_second']:.2f} | "
            f"{q4_aggregate['draft_acceptance']:.2%} | "
            f"{q4_aggregate['passed']}/{q4_aggregate['total']} | "
            f"{q4_aggregate['total']}/{q4_aggregate['total']} | "
            f"{q4_aggregate['total']}/{q4_aggregate['total']} | "
            f"{q4_aggregate['minimum_case_decode_tokens_per_second']:.2f} |"
        ),
        (
            f"| Q6 UD-Q6_K_XL | {q6_aggregate['decode_tokens_per_second']:.2f} | "
            f"{speed_ratio:.2f}x | {q6_aggregate['prompt_tokens_per_second']:.2f} | "
            f"{q6_aggregate['draft_acceptance']:.2%} | "
            f"{q6_aggregate['passed']}/{q6_total} | "
            f"{state['q6_mtp7']['exact_matches_with_q4']}/{q6_total} | "
            f"{state['q6_mtp7']['semantic_matches_with_q4']}/{q6_total} | "
            f"{q6_aggregate['minimum_case_decode_tokens_per_second']:.2f} |"
        ),
        "",
        "## By context",
        "",
        "| Quant | Context | Decode tok/s | Prompt tok/s | Acceptance |",
        "|---|---:|---:|---:|---:|",
    ]
    for quant, mode in (("Q4", q4_mode), ("Q6", state["q6_mtp7"])):
        for context, _ in sweep.CONTEXT_TARGETS:
            aggregate = mode["by_context"][context]
            lines.append(
                f"| {quant} | {context} | "
                f"{aggregate['decode_tokens_per_second']:.2f} | "
                f"{aggregate['prompt_tokens_per_second']:.2f} | "
                f"{aggregate['draft_acceptance']:.2%} |"
            )
    lines.extend([
        "",
        "## Comparison",
        "",
        f"- Q6 decode retention vs Q4: **{speed_ratio:.2%}**.",
        f"- Q6 prompt-processing retention vs Q4: **{prompt_ratio:.2%}**.",
        f"- Q6 objective quality: **{q6_aggregate['passed']}/{q6_total}**.",
        (
            "- Q6 semantic matches with Q4: "
            f"**{state['q6_mtp7']['semantic_matches_with_q4']}/{q6_total}**."
        ),
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def finalize(state: dict, q4_mode: dict) -> None:
    q6_cases = state["q6_mtp7"]["cases"]
    q4_cases = {item["id"]: item for item in q4_mode["cases"]}
    q6_mode = state["q6_mtp7"]
    q6_mode["aggregate"] = sweep.aggregate_cases(q6_cases)
    q6_mode["by_context"] = aggregate_by_context(q6_cases)
    q6_mode["exact_matches_with_q4"] = sum(
        item["content_sha256"] == q4_cases[item["id"]]["content_sha256"]
        for item in q6_cases
    )
    q6_mode["semantic_matches_with_q4"] = sum(
        sweep.semantic_signature(item["workload"], item["content"])
        == sweep.semantic_signature(
            q4_cases[item["id"]]["workload"], q4_cases[item["id"]]["content"]
        )
        for item in q6_cases
    )
    state["completed_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    save(state)
    write_report(state, q4_mode)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    q4_state = json.loads(Q4_RESULTS_PATH.read_text(encoding="utf-8"))
    q4_mode = q4_state["modes"]["mtp7"]
    q4_cases = {item["id"]: item for item in q4_mode["cases"]}
    total_cases = len(sweep.CONTEXT_TARGETS) * len(sweep.workload_definitions())

    if RESULTS_PATH.exists():
        state = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "comparison": "Qwen3.8-27B UD-Q4_K_XL vs UD-Q6_K_XL, MTP7",
            "q4_results_source": str(Q4_RESULTS_PATH),
            "q6_model_path": str(Q6_MODEL),
            "ctx_size": sweep.CTX_SIZE,
            "started_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "q6_mtp7": {"depth": 7, "cases": []},
        }

    completed_ids = {item["id"] for item in state["q6_mtp7"]["cases"]}
    if len(completed_ids) < total_cases:
        sweep.MODEL = Q6_MODEL
        sweep.PORT = PORT
        with LOG_PATH.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                sweep.server_command(7),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                sweep.wait_ready(process, LOG_PATH)
                print(
                    f"SERVER_READY q6_mtp7 rss_mib={sweep.num(sweep.rss_mib(process.pid), 1)}",
                    flush=True,
                )
                sweep.warm_up()
                cases = sweep.build_cases()
                for case in cases:
                    reference = q4_cases[case["id"]]
                    if case["prompt_sha256"] != reference["prompt_sha256"]:
                        raise RuntimeError(
                            f"prompt hash differs from Q4 for {case['id']}"
                        )
                    print(
                        f"PROMPT_VERIFIED {case['id']} tokens={case['raw_prompt_tokens']}",
                        flush=True,
                    )
                for case in cases:
                    if case["id"] in completed_ids:
                        continue
                    print(f"CASE_START q6_mtp7 {case['id']}", flush=True)
                    result = sweep.run_case(case, process)
                    state["q6_mtp7"]["cases"].append(result)
                    save(state)
                    timings = result["timings"]
                    print(
                        f"CASE_DONE q6_mtp7 {case['id']} "
                        f"pass={result['score']['passed']} "
                        f"decode_tps={timings.get('predicted_per_second', 0):.2f} "
                        f"accept={timings.get('draft_n_accepted', 0)}/"
                        f"{timings.get('draft_n', 0)}",
                        flush=True,
                    )
            finally:
                sweep.stop_server(process)

    finalize(state, q4_mode)
    aggregate = state["q6_mtp7"]["aggregate"]
    summary = {
        "q4_decode_tps": q4_mode["aggregate"]["decode_tokens_per_second"],
        "q6_decode_tps": aggregate["decode_tokens_per_second"],
        "q6_speed_ratio_vs_q4": (
            aggregate["decode_tokens_per_second"]
            / q4_mode["aggregate"]["decode_tokens_per_second"]
        ),
        "q6_prompt_tps": aggregate["prompt_tokens_per_second"],
        "q6_acceptance": aggregate["draft_acceptance"],
        "q6_quality": f"{aggregate['passed']}/{aggregate['total']}",
        "q6_exact_vs_q4": state["q6_mtp7"]["exact_matches_with_q4"],
        "q6_semantic_vs_q4": state["q6_mtp7"]["semantic_matches_with_q4"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
