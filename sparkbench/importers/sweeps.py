"""Import ``mtp_sweep/results.json`` and ``q4_q6_compare/results.json``.

Both files hold the predictable 8K/32K/64K workload (9 cases per mode:
sequence, JSON, code at three context sizes). Each mode becomes one v1 file
with a single suite, ``predictable_9``.
"""

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
    sha_from_blob_path,
)
from sparkbench.jsonutil import read_json
from sparkbench.suites.common import JSONDict, base_case, verdict_of
from sparkbench.v1 import build_document, make_suite

SUITE = "predictable_9"
DESCRIPTION = "Predictable 8K/32K/64K sweep: sequence, strict JSON and Python code at three context sizes"
SWEEP_PORT = 18082
MODEL_ALIAS = {"q4": "qwen3.8-MTP:27b", "q6": "qwen3.8-MTP:27b-q6"}


def convert_case(case: JSONDict) -> JSONDict:
    response = pseudo_response(
        content=case.get("content"),
        timings=case.get("timings"),
        wall_seconds=case.get("wall_seconds"),
        finish_reason=case.get("finish_reason"),
        usage=case.get("usage"),
    )
    score = case.get("score") or {}
    record = base_case(
        case["id"],
        response,
        verdict=verdict_of(score.get("passed")),
        prompt_sha256=case.get("prompt_sha256"),
    )
    record["detail"] = {
        "context": case.get("context"),
        "workload": case.get("workload"),
        "target_prompt_tokens": case.get("target_prompt_tokens"),
        "raw_prompt_tokens": case.get("raw_prompt_tokens"),
        "server_rss_mib": case.get("server_rss_mib"),
        "score": score,
    }
    return record


def mode_suite(mode_state: JSONDict) -> JSONDict:
    cases = [convert_case(case) for case in mode_state["cases"]]
    legacy = mode_state.get("aggregate", {})
    total = legacy.get("total") or len(cases)
    metrics = {
        "passed": legacy.get("passed"),
        "total": legacy.get("total"),
        "legacy_aggregate": legacy,
        "by_context": mode_state.get("by_context"),
    }
    for key, value in mode_state.items():
        if key.startswith(("exact_matches_with_", "semantic_matches_with_")):
            metrics[key] = value
    return make_suite(
        cases,
        description=DESCRIPTION,
        quality=(legacy.get("passed", 0) / total) if total else None,
        quality_label="exact",
        metrics=metrics,
    )


def _document(
    *,
    label: str,
    quant: str,
    depth: int | None,
    ctx_size: int,
    model_file: str | None,
    server: str | None,
    suite: JSONDict,
    source_files: list[JSONDict],
    notes: list[str],
    started: str | None,
    completed: str | None,
    extra: JSONDict,
) -> JSONDict:
    alias = MODEL_ALIAS[quant]
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
        "run": {"label": label, "quant": quant},
        "model": model_block(alias, model_file, sha_from_blob_path(model_file), QUANTIZATION[quant]),
        "runtime": {"name": "llama.cpp", "binary": server},
        "backend": {"base_url": f"http://127.0.0.1:{SWEEP_PORT}", "tokenizer": "llama.cpp", "extra_body": {"cache_prompt": False}},
        "server": {"command": [server or "llama-server", "--model", model_file or "", "--alias", alias, *llama_server_flags(ctx_size, depth, SWEEP_PORT)]},
        "request": {"temperature": 0, "seed": 42},
        "parameters": {k: v for k, v in parameters.items() if k not in {"temperature", "seed"} and v is not None},
        "workload": SUITE,
    }
    config_sha, workload_sha = config_hashes(config)
    document = build_document(
        label=label,
        quant=quant,
        model=config["model"],
        runtime=legacy_runtime(server),
        backend=legacy_backend(SWEEP_PORT),
        host=dict(LEGACY_HOST),
        parameters=parameters,
        config_sha256=config_sha,
        workload_sha256=workload_sha,
        config=config,
        source={"kind": "import", "importer": "sweeps", "files": source_files, "notes": notes},
        timings={"started_at_utc": started, "completed_at_utc": completed, "wall_seconds": suite["aggregate"]["wall_seconds"]},
        suites={SUITE: suite},
    )
    document["extra"] = extra
    return document


def import_mtp_sweep(path: Path, root: Path) -> dict[str, JSONDict]:
    state = read_json(path)
    documents: dict[str, JSONDict] = {}
    for mode, mode_state in state["modes"].items():
        depth = mode_state.get("depth")
        documents[f"mtp_sweep_q4_{mode}"] = _document(
            label=f"sweep-q4-{mode}",
            quant="q4",
            depth=depth,
            ctx_size=state["ctx_size"],
            model_file=state.get("model_path"),
            server=state.get("server"),
            suite=mode_suite(mode_state),
            source_files=[file_record(path, root)],
            notes=[f"mode {mode} of the Q4 MTP depth sweep ({state.get('model')})"],
            started=state.get("started_at_utc"),
            completed=state.get("completed_at_utc"),
            extra={"series": "mtp_sweep", "mode": mode, "sweep_winner": state.get("winner")},
        )
    return documents


def import_q4_q6_compare(path: Path, root: Path) -> dict[str, JSONDict]:
    state = read_json(path)
    mode_state = state["q6_mtp7"]
    document = _document(
        label="compare-q6-mtp7",
        quant="q6",
        depth=mode_state.get("depth"),
        ctx_size=state["ctx_size"],
        model_file=state.get("q6_model_path"),
        server=None,
        suite=mode_suite(mode_state),
        source_files=[file_record(path, root)],
        notes=[
            str(state.get("comparison")),
            "Q4 side of this comparison is mtp_sweep mode mtp7 (results/v1/mtp_sweep_q4_mtp7.json)",
        ],
        started=state.get("started_at_utc"),
        completed=state.get("completed_at_utc"),
        extra={"series": "q4_q6_compare", "mode": "q6_mtp7", "q4_results_source": state.get("q4_results_source")},
    )
    return {"q4_q6_compare_q6_mtp7": document}
