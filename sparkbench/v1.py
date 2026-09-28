"""Build ``sparkbench.result.v1`` documents (see ``schemas/result.v1.json``)."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from sparkbench import SCHEMA_ID, __version__
from sparkbench.suites import REGISTRY

JSONDict = dict[str, Any]

AGGREGATE_KEYS = (
    "cases",
    "prompt_tokens",
    "prompt_tokens_per_second",
    "generated_tokens",
    "decode_tokens_per_second",
    "median_case_decode_tokens_per_second",
    "wall_seconds",
    "draft_generated",
    "draft_accepted",
    "draft_acceptance",
)


def v1_aggregate(cases: list[JSONDict]) -> JSONDict:
    """Suite totals from v1 case records.

    Same arithmetic (and summation order) as the legacy ``aggregate()`` so
    imported results reproduce the published numbers exactly.
    """
    predicted_n = sum(int(case["tokens"].get("generated") or 0) for case in cases)
    predicted_ms = sum(float(case["time"].get("decode_ms") or 0) for case in cases)
    prompt_n = sum(int(case["tokens"].get("prompt") or 0) for case in cases)
    prompt_ms = sum(float(case["time"].get("prompt_ms") or 0) for case in cases)
    draft_n = sum(int(case["tokens"].get("draft_generated") or 0) for case in cases)
    accepted = sum(int(case["tokens"].get("draft_accepted") or 0) for case in cases)
    per_second = [float(case["tok_s"]["decode"]) for case in cases if case["tok_s"].get("decode") is not None]
    return {
        "cases": len(cases),
        "prompt_tokens": prompt_n,
        "prompt_tokens_per_second": prompt_n / (prompt_ms / 1000) if prompt_ms else None,
        "generated_tokens": predicted_n,
        "decode_tokens_per_second": predicted_n / (predicted_ms / 1000) if predicted_ms else None,
        "median_case_decode_tokens_per_second": statistics.median(per_second) if per_second else None,
        "wall_seconds": sum(float(case["time"].get("wall_seconds") or 0) for case in cases),
        "draft_generated": draft_n,
        "draft_accepted": accepted,
        "draft_acceptance": accepted / draft_n if draft_n else None,
    }


def timing_source(cases: list[JSONDict]) -> str:
    sources = {case.get("timing_source", "server") for case in cases}
    if not sources:
        return "server"
    return sources.pop() if len(sources) == 1 else "mixed"


def verdict_counts(cases: list[JSONDict]) -> JSONDict:
    counts = Counter(case["verdict"] for case in cases)
    return {key: counts.get(key, 0) for key in ("pass", "fail", "error", "unscored")}


def make_suite(
    cases: list[JSONDict],
    *,
    description: str,
    quality: float | None,
    quality_label: str,
    metrics: JSONDict,
    complete: bool = True,
    extra_aggregate: JSONDict | None = None,
) -> JSONDict:
    aggregate = v1_aggregate(cases)
    if extra_aggregate:
        aggregate.update(extra_aggregate)
    return {
        "description": description,
        "complete": complete,
        "quality": quality,
        "quality_label": quality_label,
        "verdicts": verdict_counts(cases),
        "metrics": metrics,
        "aggregate": aggregate,
        "timing_source": timing_source(cases),
        "cases": cases,
    }


def suite_from_legacy(name: str, legacy: JSONDict, complete: bool = True) -> JSONDict:
    """Convert a legacy ``run_benchmark.py`` suite state into a v1 suite."""
    spec = REGISTRY[name]
    cases = [spec.convert_case(case, legacy) for case in legacy.get("cases", [])]
    quality, label, metrics = spec.summarize(legacy)
    return make_suite(
        cases,
        description=spec.description,
        quality=quality,
        quality_label=label,
        metrics=metrics,
        complete=complete,
    )


def overall(suites: dict[str, JSONDict]) -> JSONDict:
    """Whole-workload totals, computed as in ``make_report.overall``."""
    aggregates = [suite["aggregate"] for suite in suites.values()]
    generated = sum(item["generated_tokens"] for item in aggregates)
    decode_seconds = sum(
        item["generated_tokens"] / item["decode_tokens_per_second"]
        for item in aggregates
        if item.get("decode_tokens_per_second")
    )
    draft = sum(item["draft_generated"] for item in aggregates)
    accepted = sum(item["draft_accepted"] for item in aggregates)
    return {
        "cases": sum(item["cases"] for item in aggregates),
        "api_calls": sum(int(case.get("api_calls", 1)) for suite in suites.values() for case in suite["cases"]),
        "prompt_tokens": sum(item["prompt_tokens"] for item in aggregates),
        "generated_tokens": generated,
        "decode_tokens_per_second": generated / decode_seconds if decode_seconds else None,
        "wall_seconds": sum(item["wall_seconds"] for item in aggregates),
        "draft_generated": draft,
        "draft_accepted": accepted,
        "draft_acceptance": accepted / draft if draft else None,
    }


def build_document(
    *,
    label: str,
    quant: str | None,
    model: JSONDict,
    runtime: JSONDict,
    backend: JSONDict,
    host: JSONDict,
    parameters: JSONDict,
    config_sha256: str,
    workload_sha256: str | None,
    config: JSONDict | None,
    source: JSONDict,
    timings: JSONDict,
    suites: dict[str, JSONDict],
    smoke: bool = False,
    complete: bool = True,
) -> JSONDict:
    source = {"tool": f"sparkbench {__version__}", **source}
    return {
        "schema": SCHEMA_ID,
        "label": label,
        "quant": quant,
        "model": model,
        "runtime": runtime,
        "backend": backend,
        "host": host,
        "parameters": parameters,
        "config_sha256": config_sha256,
        "workload_sha256": workload_sha256,
        "config": config,
        "source": source,
        "timings": timings,
        "smoke": smoke,
        "complete": complete,
        "suites": suites,
        "overall": overall(suites) if suites else None,
    }
