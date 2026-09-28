"""Import the partial 72-hour soak (``soak_partial/summary.json``).

``summary.json`` only has per-quant totals, so per-request records come from
the sibling ``requests.ndjson`` and run parameters from ``manifest.json``.
One v1 file per quant; one suite per request category.
"""

from __future__ import annotations

import json
from pathlib import Path

from sparkbench.importers.common import (
    LEGACY_HOST,
    QUANTIZATION,
    config_hashes,
    file_record,
    int_or_none,
    legacy_backend,
    legacy_runtime,
    model_block,
)
from sparkbench.jsonutil import read_json
from sparkbench.suites.common import JSONDict, base_case, message_content
from sparkbench.v1 import build_document, make_suite

PORT = 18084
CATEGORY_DESCRIPTIONS = {
    "deterministic": "Soak: deterministic exact-output prompts",
    "tool": "Soak: tool-call selection/serialization (positive and negative cases)",
    "reasoning": "Soak: short arithmetic reasoning with FINAL= marker",
    "code": "Soak: Python source-only function generation",
    "long8": "Soak: ~8K-token synthetic archive retrieval",
    "long32": "Soak: ~32K-token synthetic archive retrieval",
    "long60": "Soak: ~60K-token synthetic archive retrieval",
}


def convert_event(event: JSONDict) -> JSONDict:
    raw = event.get("response") or {}
    normalized = event.get("timings") or {}
    timings: JSONDict = {
        "prompt_n": int_or_none(normalized.get("prompt_n")),
        "prompt_ms": normalized.get("prompt_ms"),
        "predicted_n": int_or_none(normalized.get("predicted_n")),
        "predicted_ms": normalized.get("predicted_ms"),
        "draft_n": int_or_none(normalized.get("draft_n")),
        "draft_n_accepted": int_or_none(normalized.get("draft_n_accepted")),
        "predicted_per_second": normalized.get("decode_tokens_per_second"),
    }
    choice = (raw.get("choices") or [{}])[0]
    response = {
        "content": message_content(raw) if raw else None,
        "message": choice.get("message", {}),
        "finish_reason": choice.get("finish_reason"),
        "timings": timings,
        "wall_seconds": event.get("wall_seconds"),
    }
    verdict = ("pass" if event.get("passed") else "fail") if event.get("response_ok") else "error"
    record = base_case(
        f"{event['block_id']}/{event['request_index']}",
        response,
        verdict=verdict,
        prompt_sha256=event.get("prompt_sha256"),
    )
    record["detail"] = {
        "case_id": event.get("case_id"),
        "block_id": event.get("block_id"),
        "utc": event.get("utc"),
        "attempts": len(event.get("attempts") or []),
        "output_sha256": event.get("output_sha256"),
        "score_detail": event.get("score_detail"),
    }
    return record


def import_soak(summary_path: Path, root: Path) -> dict[str, JSONDict]:
    directory = summary_path.parent
    summary = read_json(summary_path)
    manifest_path = directory / "manifest.json"
    requests_path = directory / "requests.ndjson"
    manifest = read_json(manifest_path)
    events: list[JSONDict] = []
    with requests_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    documents: dict[str, JSONDict] = {}
    for quant, model in manifest["models"].items():
        selected = [event for event in events if event.get("quant") == quant]
        if not selected:
            continue
        suites: dict[str, JSONDict] = {}
        for category in sorted({event["category"] for event in selected}):
            cases = [convert_event(event) for event in selected if event["category"] == category]
            passed = sum(case["verdict"] == "pass" for case in cases)
            suites[category] = make_suite(
                cases,
                description=CATEGORY_DESCRIPTIONS.get(category, f"Soak: {category}"),
                quality=passed / len(cases),
                quality_label="pass rate",
                metrics={"requests": len(cases), "passed": passed},
                complete=False,
            )
        command = manifest.get(f"server_command_{quant}")
        parameters: JSONDict = {
            "mtp_depth": manifest.get("mtp_depth"),
            "temperature": 0,
            "seed": 42,
            "context": manifest.get("ctx_size"),
            "parallel": 1,
            "kv_k": "q8_0",
            "kv_v": "q8_0",
        }
        config: JSONDict = {
            "run": {"label": f"soak-{quant}-mtp{manifest.get('mtp_depth')}", "quant": quant},
            "model": model_block(model["alias"], model.get("path"), model.get("sha256"), QUANTIZATION.get(quant)),
            "runtime": {"name": "llama.cpp", "binary": manifest.get("server"), "binary_sha256": manifest.get("server_sha256")},
            "backend": {"base_url": f"http://127.0.0.1:{PORT}", "tokenizer": "llama.cpp", "extra_body": {"cache_prompt": False}},
            "server": {"command": command},
            "request": {"temperature": 0, "seed": 42},
            "parameters": {k: v for k, v in parameters.items() if k not in {"temperature", "seed"}},
            "workload": {"series": "soak", "cycle_seed": manifest.get("cycle_seed"), "shares": manifest.get("shares")},
        }
        config_sha, workload_sha = config_hashes(config)
        runtime = legacy_runtime(manifest.get("server"))
        runtime["binary_sha256"] = manifest.get("server_sha256")
        document = build_document(
            label=config["run"]["label"],
            quant=quant,
            model=config["model"],
            runtime=runtime,
            backend=legacy_backend(PORT),
            host=dict(LEGACY_HOST),
            parameters=parameters,
            config_sha256=config_sha,
            workload_sha256=workload_sha,
            config=config,
            source={
                "kind": "import",
                "importer": "soak",
                "files": [file_record(p, root) for p in (summary_path, manifest_path, requests_path)],
                "notes": [
                    "72-hour soak stopped by the operator after ~6 h 17 min; partial data",
                    "case id = <block_id>/<request_index>; verdict 'error' = request failed after retries",
                ],
            },
            timings={
                "started_at_utc": manifest.get("created_at_utc"),
                "updated_at_utc": summary.get("updated_at_utc"),
                "completed_at_utc": None,
                "wall_seconds": sum(s["aggregate"]["wall_seconds"] for s in suites.values()),
            },
            suites=suites,
            complete=False,
        )
        document["extra"] = {"series": "soak_partial", "summary": summary["by_quant"].get(quant)}
        documents[f"soak_{quant}_mtp{manifest.get('mtp_depth')}"] = document
    return documents
