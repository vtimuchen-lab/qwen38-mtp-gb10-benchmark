"""Shared suite plumbing.

The helpers below are moved from ``run_benchmark.py`` unchanged in behaviour;
only the transport (global port -> ``ctx.backend``) differs. Suites operate on
the legacy per-suite state dict (``{"cases": [...], "score": ...}``) so the
scoring code stays byte-for-byte identical; ``sparkbench.v1`` converts that
state into ``sparkbench.result.v1`` records.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparkbench.backend import ChatBackend

JSONDict = dict[str, Any]
Executor = Callable[[str, float], tuple[bool, str]]


@dataclass
class SuiteContext:
    backend: ChatBackend
    model: str
    smoke: bool = False
    seed: int = 42
    data_dir: Path = Path("data")
    humaneval_path: Path = Path("vendor/human-eval/data/HumanEval.jsonl.gz")
    ifeval_lib: Path = Path("vendor/google-research")
    options: JSONDict = field(default_factory=dict)
    checkpoint: Callable[[], None] = lambda: None
    log: Callable[[str], None] = lambda message: print(message, flush=True)
    executor: Executor | None = None


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    run: Callable[[JSONDict, SuiteContext], None]
    convert_case: Callable[[JSONDict, JSONDict], JSONDict]
    summarize: Callable[[JSONDict], tuple[float | None, str, JSONDict]]
    description: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def message_content(response: JSONDict) -> str:
    message = response.get("choices", [{}])[0].get("message", {})
    return str(message.get("content") or message.get("reasoning_content") or "")


def fallback_timings(response: JSONDict, wall_seconds: float) -> JSONDict:
    """Timings for servers without llama.cpp's ``timings`` block (vLLM, SGLang).

    Non-streaming requests expose no time-to-first-token, so decode time is
    the whole request wall time (includes prefill). Marked ``source: wall``.
    """
    usage = response.get("usage") or {}
    generated = int(usage.get("completion_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    timings: JSONDict = {
        "source": "wall",
        "prompt_n": prompt,
        "predicted_n": generated,
        "predicted_ms": wall_seconds * 1000,
    }
    if wall_seconds > 0:
        timings["predicted_per_second"] = generated / wall_seconds
    return timings


def compact_response(response: JSONDict, wall_seconds: float) -> JSONDict:
    choice = response.get("choices", [{}])[0]
    timings = response.get("timings")
    return {
        "content": message_content(response),
        "message": choice.get("message", {}),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "timings": timings if timings is not None else fallback_timings(response, wall_seconds),
        "wall_seconds": wall_seconds,
    }


def merge_responses(reasoning: JSONDict, final: JSONDict) -> JSONDict:
    """Represent a two-stage answer while accounting for all benchmark work."""
    merged = dict(final)
    first = reasoning.get("timings", {})
    second = final.get("timings", {})
    timings: dict[str, Any] = {}
    for key in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted"):
        timings[key] = (first.get(key, 0) or 0) + (second.get(key, 0) or 0)
    if timings["predicted_ms"]:
        timings["predicted_per_second"] = timings["predicted_n"] / (timings["predicted_ms"] / 1000)
    if first.get("source") == "wall" or second.get("source") == "wall":
        timings["source"] = "wall"
    merged["timings"] = timings
    merged["wall_seconds"] = reasoning.get("wall_seconds", 0) + final.get("wall_seconds", 0)
    return merged


def aggregate(items: list[JSONDict]) -> JSONDict:
    timings = [item.get("response", {}).get("timings", {}) for item in items]
    predicted_n = sum(int(item.get("predicted_n", 0) or 0) for item in timings)
    predicted_ms = sum(float(item.get("predicted_ms", 0) or 0) for item in timings)
    prompt_n = sum(int(item.get("prompt_n", 0) or 0) for item in timings)
    prompt_ms = sum(float(item.get("prompt_ms", 0) or 0) for item in timings)
    draft_n = sum(int(item.get("draft_n", 0) or 0) for item in timings)
    accepted = sum(int(item.get("draft_n_accepted", 0) or 0) for item in timings)
    per_second = [
        float(item["predicted_per_second"])
        for item in timings
        if item.get("predicted_per_second") is not None
    ]
    return {
        "cases": len(items),
        "prompt_tokens": prompt_n,
        "prompt_tokens_per_second": prompt_n / (prompt_ms / 1000) if prompt_ms else None,
        "generated_tokens": predicted_n,
        "decode_tokens_per_second": predicted_n / (predicted_ms / 1000) if predicted_ms else None,
        "median_case_decode_tokens_per_second": statistics.median(per_second) if per_second else None,
        "wall_seconds": sum(float(item.get("response", {}).get("wall_seconds", 0)) for item in items),
        "draft_generated": draft_n,
        "draft_accepted": accepted,
        "draft_acceptance": accepted / draft_n if draft_n else None,
    }


# -- v1 case helpers -------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def normalize_tool_calls(message: JSONDict) -> list[JSONDict] | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    normalized: list[JSONDict] = []
    for call in calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        arguments: Any = function.get("arguments")
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        normalized.append({"name": function.get("name"), "arguments": arguments})
    return normalized


def verdict_of(value: bool | None) -> str:
    if value is None:
        return "unscored"
    return "pass" if value else "fail"


def base_case(
    case_id: str,
    response: JSONDict,
    *,
    verdict: str,
    prompt_sha256: str | None,
    api_calls: int = 1,
    transcript: list[str] | None = None,
) -> JSONDict:
    """Common per-case record shared by every suite (see schemas/result.v1.json)."""
    timings = response.get("timings") or {}
    content = response.get("content")
    prompt_ms = _float_or_none(timings.get("prompt_ms"))
    prompt_n = _int_or_none(timings.get("prompt_n"))
    prompt_rate = _float_or_none(timings.get("prompt_per_second"))
    if prompt_rate is None and prompt_n is not None and prompt_ms:
        prompt_rate = prompt_n / (prompt_ms / 1000)
    record: JSONDict = {
        "id": str(case_id),
        "verdict": verdict,
        "prompt_sha256": prompt_sha256,
        "api_calls": api_calls,
        "tokens": {
            "prompt": prompt_n,
            "generated": _int_or_none(timings.get("predicted_n")),
            "draft_generated": _int_or_none(timings.get("draft_n")),
            "draft_accepted": _int_or_none(timings.get("draft_n_accepted")),
        },
        "time": {
            "wall_seconds": _float_or_none(response.get("wall_seconds")),
            "prompt_ms": prompt_ms,
            "decode_ms": _float_or_none(timings.get("predicted_ms")),
        },
        "tok_s": {
            "prompt": prompt_rate,
            "decode": _float_or_none(timings.get("predicted_per_second")),
        },
        "timing_source": str(timings.get("source", "server")),
        "finish_reason": response.get("finish_reason"),
        "content": content,
        "content_sha256": sha256_text(content) if isinstance(content, str) else None,
    }
    tool_calls = normalize_tool_calls(response.get("message") or {})
    if tool_calls is not None:
        record["tool_calls"] = tool_calls
    if transcript is not None:
        record["transcript"] = transcript
    return record
