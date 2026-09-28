"""Paired A/B report for two result.v1 files (generalised make_report.py)."""

from __future__ import annotations

from sparkbench.stats import paired
from sparkbench.suites.common import JSONDict

LABELS = {
    "mmlu_pro": "MMLU-Pro-100",
    "gsm8k": "GSM8K-100",
    "humaneval": "HumanEval-40",
    "ifeval": "IFEval-50 strict prompt",
    "long_context": "Long context 9",
    "tools": "Tool calls 20",
    "stability_100": "Stability 100",
}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def ratio_delta(numerator: float | None, denominator: float | None) -> float | None:
    if not numerator or not denominator:
        return None
    return numerator / denominator - 1


def signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def scored_map(suite: JSONDict) -> dict[str, bool] | None:
    cases = suite["cases"]
    if not cases or any(case["verdict"] not in ("pass", "fail") for case in cases):
        return None
    return {case["id"]: case["verdict"] == "pass" for case in cases}


def common_suites(a: JSONDict, b: JSONDict) -> list[str]:
    return [name for name in a["suites"] if name in b["suites"]]


def suite_row(suite: JSONDict) -> JSONDict:
    aggregate = suite["aggregate"]
    return {
        "quality": suite["quality"],
        "quality_label": suite["quality_label"],
        "passed": suite["verdicts"]["pass"],
        "scored": suite["verdicts"]["pass"] + suite["verdicts"]["fail"] + suite["verdicts"]["error"],
        "cases": aggregate["cases"],
        "decode_tokens_per_second": aggregate["decode_tokens_per_second"],
        "prompt_tokens_per_second": aggregate["prompt_tokens_per_second"],
        "draft_acceptance": aggregate["draft_acceptance"],
        "generated_tokens": aggregate["generated_tokens"],
    }


def build_summary(a: JSONDict, b: JSONDict) -> JSONDict:
    names = common_suites(a, b)
    summary: JSONDict = {
        "a": {"label": a["label"], "overall": a["overall"], "suites": {n: suite_row(a["suites"][n]) for n in names}},
        "b": {"label": b["label"], "overall": b["overall"], "suites": {n: suite_row(b["suites"][n]) for n in names}},
        "speed": {},
        "paired": {},
        "warnings": [],
    }
    if a.get("workload_sha256") and a.get("workload_sha256") == b.get("workload_sha256"):
        summary["warnings"].append("workload_sha256 is identical: A and B measure the same model and settings")
    for name in names:
        left = a["suites"][name]["aggregate"]["decode_tokens_per_second"]
        right = b["suites"][name]["aggregate"]["decode_tokens_per_second"]
        summary["speed"][name] = {"b_vs_a": ratio_delta(right, left), "a_vs_b": ratio_delta(left, right)}
        left_map = scored_map(a["suites"][name])
        right_map = scored_map(b["suites"][name])
        if left_map is None or right_map is None:
            continue
        if left_map.keys() != right_map.keys():
            summary["warnings"].append(f"{name}: case ids differ, paired statistics skipped")
            continue
        summary["paired"][name] = paired(left_map, right_map)
    left_all = a["overall"]["decode_tokens_per_second"]
    right_all = b["overall"]["decode_tokens_per_second"]
    summary["speed"]["overall"] = {"b_vs_a": ratio_delta(right_all, left_all), "a_vs_b": ratio_delta(left_all, right_all)}
    if "mmlu_pro" in names:
        left_cats = a["suites"]["mmlu_pro"]["metrics"].get("by_category", {})
        right_cats = b["suites"]["mmlu_pro"]["metrics"].get("by_category", {})
        summary["mmlu_categories"] = {cat: {"a": score, "b": right_cats.get(cat)} for cat, score in left_cats.items()}
    return summary


def _count(suite: JSONDict) -> str:
    verdicts = suite["verdicts"]
    total = verdicts["pass"] + verdicts["fail"] + verdicts["error"]
    return f"{verdicts['pass']}/{total}"


def render_markdown(a: JSONDict, b: JSONDict, summary: JSONDict) -> str:
    la, lb = a["label"], b["label"]
    names = common_suites(a, b)
    lines = [
        f"# {la} vs {lb}",
        "",
        f"- A: `{la}` — model `{a['model']['name']}`, runtime {a['runtime'].get('name')} {a['runtime'].get('version') or ''}, host {a['host'].get('name') or 'n/a'} ({a['host'].get('gpu') or 'unknown GPU'})".rstrip(),
        f"- B: `{lb}` — model `{b['model']['name']}`, runtime {b['runtime'].get('name')} {b['runtime'].get('version') or ''}, host {b['host'].get('name') or 'n/a'} ({b['host'].get('gpu') or 'unknown GPU'})".rstrip(),
        "",
        "## Main result",
        "",
        f"| Test | {la} quality | {lb} quality | {la} tok/s | {lb} tok/s | {lb} speed delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        left = summary["a"]["suites"][name]
        right = summary["b"]["suites"][name]
        lines.append(
            f"| {LABELS.get(name, name)} | {pct(left['quality'])} | {pct(right['quality'])} | "
            f"{num(left['decode_tokens_per_second'])} | {num(right['decode_tokens_per_second'])} | "
            f"{signed(summary['speed'][name]['b_vs_a'])} |"
        )
    if summary["paired"]:
        lines += [
            "",
            "## Paired correctness",
            "",
            f"| Test | Both pass | {la} only | {lb} only | Both fail | Exact McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, item in summary["paired"].items():
            lines.append(
                f"| {name} | {item['both_correct']} | {item['a_only']} | {item['b_only']} | "
                f"{item['both_wrong']} | {item['mcnemar_exact_p']:.4f} |"
            )
    if "ifeval" in names:
        ma = a["suites"]["ifeval"]["metrics"]
        mb = b["suites"]["ifeval"]["metrics"]
        if "strict" in ma and "strict" in mb:
            lines += ["", "## IFEval detail", "", f"| Metric | {la} | {lb} |", "|---|---:|---:|"]
            for mode in ("strict", "loose"):
                for level in ("prompt", "instruction"):
                    key = f"{level}_level"
                    lines.append(f"| {mode.title()} {level} | {pct(ma[mode][key])} | {pct(mb[mode][key])} |")
    exact_suites = [n for n in ("long_context", "tools", "stability_100") if n in names]
    if exact_suites:
        lines += ["", "## Long context, tools and stability", ""]
        if "long_context" in names:
            lines.append(f"- Long-context exact: {la} {_count(a['suites']['long_context'])}, {lb} {_count(b['suites']['long_context'])}.")
            lines.append(
                f"- Long-context prefill: {la} {num(summary['a']['suites']['long_context']['prompt_tokens_per_second'])} tok/s, "
                f"{lb} {num(summary['b']['suites']['long_context']['prompt_tokens_per_second'])} tok/s."
            )
        if "tools" in names:
            lines.append(f"- Tool calls: {la} {_count(a['suites']['tools'])}, {lb} {_count(b['suites']['tools'])}.")
        if "stability_100" in names:
            ua = a["suites"]["stability_100"]["metrics"].get("unique_outputs")
            ub = b["suites"]["stability_100"]["metrics"].get("unique_outputs")
            lines.append(
                f"- Repeated prompts: {la} {_count(a['suites']['stability_100'])} exact ({ua} unique output(s)), "
                f"{lb} {_count(b['suites']['stability_100'])} exact ({ub} unique output(s))."
            )
    oa, ob = a["overall"], b["overall"]
    faster = summary["speed"]["overall"]["a_vs_b"]
    lines += [
        "",
        "## Overall workload",
        "",
        f"- {la}: {oa['cases']} scored cases, {oa['api_calls']} API calls, {oa['generated_tokens']} generated tokens at weighted {num(oa['decode_tokens_per_second'])} tok/s, MTP/draft acceptance {pct(oa['draft_acceptance'])}.",
        f"- {lb}: {ob['cases']} scored cases, {ob['api_calls']} API calls, {ob['generated_tokens']} generated tokens at weighted {num(ob['decode_tokens_per_second'])} tok/s, MTP/draft acceptance {pct(ob['draft_acceptance'])}.",
    ]
    if faster is not None:
        word = "faster" if faster >= 0 else "slower"
        lines.append(f"- {la} is {abs(faster):.1%} {word} than {lb} ({lb} as denominator).")
    if summary.get("mmlu_categories"):
        lines += ["", "## MMLU-Pro categories", "", f"| Category | {la} | {lb} |", "|---|---:|---:|"]
        for category, item in summary["mmlu_categories"].items():
            lines.append(f"| {category} | {pct(item['a'])} | {pct(item['b'])} |")
    if summary["warnings"]:
        lines += ["", "## Notes", ""] + [f"- {warning}" for warning in summary["warnings"]]
    return "\n".join(lines) + "\n"


def report(a: JSONDict, b: JSONDict) -> tuple[JSONDict, str]:
    summary = build_summary(a, b)
    return summary, render_markdown(a, b, summary)
