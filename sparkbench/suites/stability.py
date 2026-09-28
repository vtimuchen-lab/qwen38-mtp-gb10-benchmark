"""Stability: 100 identical deterministic prompts."""

from __future__ import annotations

import hashlib
from collections import Counter

from sparkbench.suites.common import (
    JSONDict,
    SuiteContext,
    SuiteSpec,
    base_case,
    compact_response,
    sha256_text,
    verdict_of,
)
from sparkbench.suites.common import aggregate as aggregate_cases

PROMPT = "Return exactly this text and nothing else: BENCHMARK_OK_42"
EXPECTED = "BENCHMARK_OK_42"


def run(suite: JSONDict, ctx: SuiteContext) -> None:
    count = 3 if ctx.smoke else 100
    suite.setdefault("cases", [])
    done = {item["id"] for item in suite["cases"]}
    for index in range(count):
        case_id = str(index)
        if case_id in done:
            continue
        response, wall = ctx.backend.chat(ctx.model, [{"role": "user", "content": PROMPT}], 32)
        compact = compact_response(response, wall)
        content = compact["content"].strip()
        suite["cases"].append({"id": case_id, "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "exact": content == EXPECTED, "response": compact})
        ctx.checkpoint()
        if (index + 1) % 10 == 0 or ctx.smoke:
            ctx.log(f"Stability {index + 1}/{count}")
    finalize(suite)
    ctx.checkpoint()


def finalize(suite: JSONDict) -> None:
    cases = suite["cases"]
    counts = Counter(item["content_sha256"] for item in cases)
    suite["exact_rate"] = sum(item["exact"] for item in cases) / len(cases)
    suite["unique_outputs"] = len(counts)
    suite["modal_output_rate"] = max(counts.values()) / len(cases)
    suite["aggregate"] = aggregate_cases(cases)


def convert_case(case: JSONDict, suite: JSONDict) -> JSONDict:
    record = base_case(
        case["id"],
        case["response"],
        verdict=verdict_of(case.get("exact")),
        prompt_sha256=sha256_text(PROMPT),
    )
    record["expected"] = EXPECTED
    record["detail"] = {"stripped_content_sha256": case.get("content_sha256")}
    return record


def summarize(suite: JSONDict) -> tuple[float | None, str, JSONDict]:
    metrics = {key: suite.get(key) for key in ("exact_rate", "unique_outputs", "modal_output_rate")}
    return suite.get("exact_rate"), "exact", metrics


SPEC = SuiteSpec("stability_100", run, convert_case, summarize, "100 identical prompts")
