"""Benchmark suites moved from ``run_benchmark.py`` (scoring unchanged)."""

from __future__ import annotations

from sparkbench.suites import gsm8k, humaneval, ifeval, long_context, mmlu_pro, stability, tools
from sparkbench.suites.common import SuiteContext, SuiteSpec

REGISTRY: dict[str, SuiteSpec] = {
    spec.name: spec
    for spec in (
        mmlu_pro.SPEC,
        gsm8k.SPEC,
        humaneval.SPEC,
        ifeval.SPEC,
        long_context.SPEC,
        tools.SPEC,
        stability.SPEC,
    )
}

__all__ = ["REGISTRY", "SuiteContext", "SuiteSpec"]
