"""Same config on two machines: throughput deltas and answer identity.

With temperature 0 and a fixed seed, the same model, runtime build and
settings should produce the same answers on two hosts. This module reports

* decode/prefill tok/s per suite and overall, plus the per-case speed ratio;
* which paired cases have different outputs (content, tool calls, earlier
  turns), with differences in the *numbers* inside answers listed separately
  because those change results, while wording drift often does not;
* verdict flips and cases present on only one side.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from sparkbench.report import num, pct, ratio_delta, signed
from sparkbench.suites.common import JSONDict

NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")


def numbers(text: str | None) -> list[str]:
    return NUMBER.findall(text or "")


def _answer(case: JSONDict) -> tuple[Any, ...]:
    return (case.get("content"), case.get("tool_calls"), case.get("transcript"))


def _answer_text(case: JSONDict) -> str:
    parts = list(case.get("transcript") or [])
    parts.append(case.get("content") or "")
    for call in case.get("tool_calls") or []:
        parts.append(repr(call))
    return "\n".join(parts)


def _first_difference(left: str, right: str) -> int:
    for index, (x, y) in enumerate(zip(left, right, strict=False)):
        if x != y:
            return index
    return min(len(left), len(right))


def _snippet(text: str, at: int, width: int = 40) -> str:
    start = max(0, at - width // 2)
    return text[start : start + width].replace("\n", "\\n")


def compare_suite(name: str, left: JSONDict, right: JSONDict) -> JSONDict:
    lcases = {case["id"]: case for case in left["cases"]}
    rcases = {case["id"]: case for case in right["cases"]}
    common = [case_id for case_id in lcases if case_id in rcases]
    identical = 0
    numeric: list[JSONDict] = []
    textual: list[JSONDict] = []
    flips: list[JSONDict] = []
    ratios: list[float] = []
    for case_id in common:
        a, b = lcases[case_id], rcases[case_id]
        ra, rb = a["tok_s"].get("decode"), b["tok_s"].get("decode")
        if ra and rb:
            ratios.append(rb / ra)
        if a["verdict"] != b["verdict"]:
            flips.append({"suite": name, "id": case_id, "a": a["verdict"], "b": b["verdict"]})
        if _answer(a) == _answer(b):
            identical += 1
            continue
        text_a, text_b = _answer_text(a), _answer_text(b)
        na, nb = numbers(text_a), numbers(text_b)
        at = _first_difference(text_a, text_b)
        entry: JSONDict = {
            "suite": name,
            "id": case_id,
            "first_difference_at": at,
            "a_excerpt": _snippet(text_a, at),
            "b_excerpt": _snippet(text_b, at),
        }
        if na != nb:
            only_a = [x for x in na if x not in nb]
            only_b = [x for x in nb if x not in na]
            entry.update({"a_numbers_only": only_a, "b_numbers_only": only_b, "a_final_number": na[-1] if na else None, "b_final_number": nb[-1] if nb else None})
            numeric.append(entry)
        else:
            textual.append(entry)
    la, ra_ = left["aggregate"], right["aggregate"]
    return {
        "cases_a": len(lcases),
        "cases_b": len(rcases),
        "paired": len(common),
        "only_a": [case_id for case_id in lcases if case_id not in rcases],
        "only_b": [case_id for case_id in rcases if case_id not in lcases],
        "identical": identical,
        "numeric_differences": numeric,
        "text_differences": textual,
        "verdict_flips": flips,
        "quality_a": left["quality"],
        "quality_b": right["quality"],
        "decode_a": la["decode_tokens_per_second"],
        "decode_b": ra_["decode_tokens_per_second"],
        "decode_b_vs_a": ratio_delta(ra_["decode_tokens_per_second"], la["decode_tokens_per_second"]),
        "prefill_a": la["prompt_tokens_per_second"],
        "prefill_b": ra_["prompt_tokens_per_second"],
        "prefill_b_vs_a": ratio_delta(ra_["prompt_tokens_per_second"], la["prompt_tokens_per_second"]),
        "median_case_ratio_b_over_a": statistics.median(ratios) if ratios else None,
        "cases_b_faster": sum(ratio > 1 for ratio in ratios),
        "timing_source_a": left["timing_source"],
        "timing_source_b": right["timing_source"],
    }


def compare(a: JSONDict, b: JSONDict) -> JSONDict:
    checks: list[str] = []
    if a.get("workload_sha256") != b.get("workload_sha256"):
        checks.append("workload_sha256 differs: the two runs did not use the same model/settings; differences are expected")
    if a.get("config_sha256") != b.get("config_sha256"):
        checks.append("config_sha256 differs (host-specific paths/URLs may legitimately differ)")
    if a["model"].get("sha256") != b["model"].get("sha256"):
        checks.append("model sha256 differs")
    if a["runtime"].get("version") != b["runtime"].get("version"):
        checks.append(f"runtime version differs: {a['runtime'].get('version')} vs {b['runtime'].get('version')}")
    if a["host"].get("name") and a["host"].get("name") == b["host"].get("name"):
        checks.append("host name is identical on both sides")
    suites = {name: compare_suite(name, a["suites"][name], b["suites"][name]) for name in a["suites"] if name in b["suites"]}
    missing = sorted(set(a["suites"]) ^ set(b["suites"]))
    oa, ob = a["overall"] or {}, b["overall"] or {}
    return {
        "a": {"label": a["label"], "host": a["host"], "config_sha256": a["config_sha256"], "workload_sha256": a.get("workload_sha256")},
        "b": {"label": b["label"], "host": b["host"], "config_sha256": b["config_sha256"], "workload_sha256": b.get("workload_sha256")},
        "checks": checks,
        "suites_only_on_one_side": missing,
        "suites": suites,
        "overall": {
            "decode_a": oa.get("decode_tokens_per_second"),
            "decode_b": ob.get("decode_tokens_per_second"),
            "decode_b_vs_a": ratio_delta(ob.get("decode_tokens_per_second"), oa.get("decode_tokens_per_second")),
            "paired_cases": sum(s["paired"] for s in suites.values()),
            "identical": sum(s["identical"] for s in suites.values()),
            "numeric_differences": sum(len(s["numeric_differences"]) for s in suites.values()),
            "text_differences": sum(len(s["text_differences"]) for s in suites.values()),
            "verdict_flips": sum(len(s["verdict_flips"]) for s in suites.values()),
        },
    }


def _host(side: JSONDict) -> str:
    host = side["host"]
    return f"{host.get('name') or 'unnamed host'} ({host.get('gpu') or 'unknown GPU'}, driver {host.get('driver') or 'n/a'})"


def render_markdown(result: JSONDict) -> str:
    a, b = result["a"], result["b"]
    overall = result["overall"]
    lines = [
        f"# Host comparison: {a['label']} vs {b['label']}",
        "",
        f"- A: {_host(a)}",
        f"- B: {_host(b)}",
        f"- Answers identical: {overall['identical']}/{overall['paired_cases']} paired cases; "
        f"{overall['numeric_differences']} with different numbers, {overall['text_differences']} with text-only differences, "
        f"{overall['verdict_flips']} verdict flips.",
        f"- Overall decode: A {num(overall['decode_a'])} tok/s, B {num(overall['decode_b'])} tok/s ({signed(overall['decode_b_vs_a'])} B vs A).",
    ]
    if result["checks"]:
        lines += ["", "## Checks", ""] + [f"- {item}" for item in result["checks"]]
    lines += [
        "",
        "## Throughput by suite",
        "",
        "| Suite | Cases | A decode tok/s | B decode tok/s | B vs A | Median per-case B/A | B faster cases | A prefill tok/s | B prefill tok/s | B vs A prefill |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["suites"].items():
        median = item["median_case_ratio_b_over_a"]
        lines.append(
            f"| {name} | {item['paired']} | {num(item['decode_a'])} | {num(item['decode_b'])} | {signed(item['decode_b_vs_a'])} | "
            f"{'n/a' if median is None else f'{median:.3f}'} | {item['cases_b_faster']}/{item['paired']} | "
            f"{num(item['prefill_a'])} | {num(item['prefill_b'])} | {signed(item['prefill_b_vs_a'])} |"
        )
    lines += [
        "",
        "## Answer identity by suite",
        "",
        "| Suite | Paired | Identical | Numeric diffs | Text-only diffs | Verdict flips | A quality | B quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["suites"].items():
        lines.append(
            f"| {name} | {item['paired']} | {item['identical']} | {len(item['numeric_differences'])} | "
            f"{len(item['text_differences'])} | {len(item['verdict_flips'])} | {pct(item['quality_a'])} | {pct(item['quality_b'])} |"
        )
    numeric = [entry for item in result["suites"].values() for entry in item["numeric_differences"]]
    lines += ["", "## Differences in numbers", ""]
    if numeric:
        lines += ["| Suite | Case | Numbers only in A | Numbers only in B | Final number A | Final number B |", "|---|---|---|---|---:|---:|"]
        for entry in numeric:
            lines.append(
                f"| {entry['suite']} | `{entry['id']}` | {', '.join(entry['a_numbers_only'][:8]) or '-'} | "
                f"{', '.join(entry['b_numbers_only'][:8]) or '-'} | {entry['a_final_number']} | {entry['b_final_number']} |"
            )
    else:
        lines.append("None.")
    textual = [entry for item in result["suites"].values() for entry in item["text_differences"]]
    lines += ["", "## Text-only differences", ""]
    if textual:
        for entry in textual:
            lines.append(f"- {entry['suite']} `{entry['id']}` at char {entry['first_difference_at']}: A `{entry['a_excerpt']}` / B `{entry['b_excerpt']}`")
    else:
        lines.append("None.")
    flips = [entry for item in result["suites"].values() for entry in item["verdict_flips"]]
    if flips:
        lines += ["", "## Verdict flips", ""] + [f"- {f['suite']} `{f['id']}`: A {f['a']}, B {f['b']}" for f in flips]
    missing_cases = [(name, item) for name, item in result["suites"].items() if item["only_a"] or item["only_b"]]
    if missing_cases or result["suites_only_on_one_side"]:
        lines += ["", "## Unpaired", ""]
        if result["suites_only_on_one_side"]:
            lines.append(f"- Suites on one side only: {', '.join(result['suites_only_on_one_side'])}")
        for name, item in missing_cases:
            lines.append(f"- {name}: {len(item['only_a'])} case(s) only in A, {len(item['only_b'])} only in B")
    return "\n".join(lines) + "\n"
