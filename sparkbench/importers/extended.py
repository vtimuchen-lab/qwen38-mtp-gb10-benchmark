"""Import ``extended_validation/*.json`` (mixed mini-sweep and prefix-cache A/B)."""

from __future__ import annotations

from pathlib import Path

from sparkbench.importers.common import (
    LEGACY_HOST,
    QUANTIZATION,
    config_hashes,
    file_record,
    legacy_backend,
    legacy_runtime,
    llama_server_flags,
    model_block,
    pseudo_response,
)
from sparkbench.jsonutil import read_json
from sparkbench.suites import REGISTRY
from sparkbench.suites.common import JSONDict, base_case, merge_responses, sha256_text, verdict_of
from sparkbench.v1 import build_document, make_suite

PORT = 18084
Q4_ALIAS = "qwen3.8-MTP:27b"
Q4_SHA = "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"

MINI_DESCRIPTIONS = {
    "ifeval": "Mixed mini-sweep: first 10 IFEval prompts of the 50-prompt sample, official strict/loose scorer",
    "gsm8k": "Mixed mini-sweep: first 10 GSM8K problems of the 100-problem sample, 8-shot CoT",
    "tools": "Mixed mini-sweep: first 10 of the 20 exact tool-call checks",
    "chat": "Mixed mini-sweep: 10 short chat prompts, non-empty and non-pathological output",
}


def _chat_case(case: JSONDict) -> JSONDict:
    record = base_case(
        case["id"],
        case["response"],
        verdict=verdict_of(case.get("valid")),
        prompt_sha256=sha256_text(case["prompt"]),
    )
    return record


def _document(
    *,
    label: str,
    depth: int | None,
    ctx_size: int,
    suites: dict[str, JSONDict],
    path: Path,
    root: Path,
    series: str,
    notes: list[str],
    timings: JSONDict,
    extra: JSONDict,
    extra_body: JSONDict,
) -> JSONDict:
    parameters: JSONDict = {
        "mtp_depth": depth,
        "temperature": 0,
        "seed": 42,
        "context": ctx_size,
        "parallel": 1,
        "kv_k": "q8_0",
        "kv_v": "q8_0",
    }
    config: JSONDict = {
        "run": {"label": label, "quant": "q4"},
        "model": model_block(Q4_ALIAS, None, Q4_SHA, QUANTIZATION["q4"]),
        "runtime": {"name": "llama.cpp"},
        "backend": {"base_url": f"http://127.0.0.1:{PORT}", "tokenizer": "llama.cpp", "extra_body": extra_body},
        "server": {"command": ["llama-server", "--alias", Q4_ALIAS, *llama_server_flags(ctx_size, depth, PORT)]},
        "request": {"temperature": 0, "seed": 42},
        "parameters": {k: v for k, v in parameters.items() if k not in {"temperature", "seed"} and v is not None},
        "workload": series,
    }
    config_sha, workload_sha = config_hashes(config)
    document = build_document(
        label=label,
        quant="q4",
        model=config["model"],
        runtime=legacy_runtime(None),
        backend=legacy_backend(PORT, extra_body),
        host=dict(LEGACY_HOST),
        parameters=parameters,
        config_sha256=config_sha,
        workload_sha256=workload_sha,
        config=config,
        source={"kind": "import", "importer": "extended", "files": [file_record(path, root)], "notes": notes},
        timings=timings,
        suites=suites,
    )
    document["extra"] = extra
    return document


def import_complex(path: Path, root: Path) -> dict[str, JSONDict]:
    state = read_json(path)
    documents: dict[str, JSONDict] = {}
    for mode, mode_state in state["modes"].items():
        suites: dict[str, JSONDict] = {}
        for name, legacy in mode_state["suites"].items():
            if name == "chat":
                cases = [_chat_case(case) for case in legacy["cases"]]
                quality, label, metrics = legacy.get("score"), "valid", {"score": legacy.get("score")}
            else:
                spec = REGISTRY[name]
                cases = [spec.convert_case(case, legacy) for case in legacy["cases"]]
                quality, label, metrics = spec.summarize(legacy)
            suites[f"mini_{name}"] = make_suite(
                cases,
                description=MINI_DESCRIPTIONS[name],
                quality=quality,
                quality_label=label,
                metrics=metrics,
                complete=bool(mode_state.get("complete")),
            )
        memory = {key: mode_state[key] for key in ("memory_ready", "memory_after_warmup", "memory_after_suite") if key in mode_state}
        documents[f"complex_q4_{mode}"] = _document(
            label=f"complex-q4-{mode}",
            depth=mode_state.get("depth"),
            ctx_size=65536,
            suites=suites,
            path=path,
            root=root,
            series="complex_mini_sweep",
            notes=[f"mode {mode} of the mixed IFEval/GSM8K/tool/chat mini-sweep ({state.get('model')})"],
            timings={
                "started_at_utc": state.get("started_at_utc"),
                "completed_at_utc": state.get("completed_at_utc"),
                "server_load_seconds": mode_state.get("load_seconds"),
                "wall_seconds": sum(s["aggregate"]["wall_seconds"] for s in suites.values()),
            },
            extra={"series": "extended_validation/complex", "mode": mode, "legacy_aggregate": mode_state.get("aggregate"), "memory": memory},
            extra_body={"cache_prompt": False},
        )
    return documents


def _prefix_case(case: JSONDict) -> JSONDict:
    tool = pseudo_response(content=None, timings=case.get("tool_timings"), wall_seconds=case.get("tool_wall_seconds"))
    final = pseudo_response(content=case.get("actual"), timings=case.get("final_timings"), wall_seconds=case.get("final_wall_seconds"))
    merged = merge_responses(tool, final)
    passed = bool(case.get("tool_ok")) and bool(case.get("final_ok"))
    record = base_case(f"turn_{case['turn']}", merged, verdict=verdict_of(passed), prompt_sha256=None, api_calls=2)
    record["tool_calls"] = [{"name": "lookup_order", "arguments": case.get("arguments")}]
    record["expected"] = case.get("expected")
    record["predicted"] = case.get("actual")
    record["detail"] = {"order_id": case.get("order_id"), "tool_ok": case.get("tool_ok"), "final_ok": case.get("final_ok")}
    return record


def import_prefix_cache(path: Path, root: Path) -> dict[str, JSONDict]:
    state = read_json(path)
    documents: dict[str, JSONDict] = {}
    for mode, mode_state in state["modes"].items():
        cases = [_prefix_case(case) for case in mode_state["cases"]]
        legacy_summary = {key: value for key, value in mode_state.items() if key != "cases"}
        total = mode_state.get("quality_total") or len(cases)
        suite = make_suite(
            cases,
            description="Progressive 12-turn agent session (~18K-token system prompt): tool call then exact final answer per turn",
            quality=mode_state.get("quality_passed", 0) / total if total else None,
            quality_label="exact",
            metrics={"legacy_summary": legacy_summary},
        )
        cache_prompt = bool(mode_state.get("cache_prompt"))
        documents[f"prefix_cache_q4_mtp7_{mode}"] = _document(
            label=f"prefix-q4-mtp7-{mode}",
            depth=7,
            ctx_size=32768,
            suites={"agent_session_12": suite},
            path=path,
            root=root,
            series="prefix_cache",
            notes=[
                "prefix-cache A/B; prompt text was not stored so prompt_sha256 is null",
                "Each case = one turn (tool call + final answer); tokens/time are the sum of both calls",
            ],
            timings={"completed_at_utc": state.get("completed_at_utc"), "wall_seconds": suite["aggregate"]["wall_seconds"]},
            extra={"series": "extended_validation/prefix_cache", "mode": mode, "comparison": state.get("comparison")},
            extra_body={"cache_prompt": cache_prompt},
        )
    return documents
