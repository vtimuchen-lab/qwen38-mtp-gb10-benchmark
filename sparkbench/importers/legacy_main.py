"""Import ``results/q4.json`` / ``results/q6.json`` (run_benchmark.py, schema 1)."""

from __future__ import annotations

from pathlib import Path

from sparkbench.config import ALL_SUITES
from sparkbench.importers.common import (
    LEGACY_HOST,
    LEGACY_RUNTIME_VERSION,
    QUANTIZATION,
    config_hashes,
    file_record,
    legacy_backend,
    legacy_runtime,
    llama_server_flags,
    model_block,
)
from sparkbench.jsonutil import read_json
from sparkbench.suites.common import JSONDict
from sparkbench.v1 import build_document, suite_from_legacy

LEGACY_PORT = 18084


def legacy_config(state: JSONDict) -> JSONDict:
    """The run_benchmark.py configuration expressed as a sparkbench TOML dict.

    ``examples/llamacpp-gb10-q4-mtp7.toml`` is this dict for Q4 written out as
    TOML; a test checks that both hash identically.
    """
    quant = state["quant"]
    params = state["parameters"]
    label = f"{quant}-mtp{params['mtp_depth']}"
    return {
        "run": {"label": label, "quant": quant, "output": f"results/runs/{label}.json"},
        "model": model_block(state["model"], state["model_file"], state["model_sha256"], QUANTIZATION.get(quant)),
        "runtime": {"name": "llama.cpp", "version": LEGACY_RUNTIME_VERSION, "binary": state["runtime"]},
        "backend": {
            "base_url": f"http://127.0.0.1:{LEGACY_PORT}",
            "tokenizer": "llama.cpp",
            "extra_body": {"cache_prompt": False},
        },
        "server": {
            "command": [
                state["runtime"],
                "--model",
                state["model_file"],
                "--alias",
                state["model"],
                *llama_server_flags(params["context"], params["mtp_depth"], LEGACY_PORT),
            ],
            "ready_timeout": 600,
        },
        "request": {"temperature": params["temperature"], "seed": params["seed"]},
        "parameters": {k: v for k, v in params.items() if k not in {"temperature", "seed"}},
        "data": {"nltk_data": "/home/admin/.local/share/qwen38-nltk-data"},
    }


def import_legacy_main(path: Path, root: Path) -> JSONDict:
    state = read_json(path)
    suites = {name: suite_from_legacy(name, state["suites"][name]) for name in ALL_SUITES if name in state["suites"]}
    config = legacy_config(state)
    config_sha, workload_sha = config_hashes(config)
    wall = sum(suite["aggregate"]["wall_seconds"] for suite in suites.values())
    document = build_document(
        label=config["run"]["label"],
        quant=state["quant"],
        model=config["model"],
        runtime=legacy_runtime(state["runtime"]),
        backend=legacy_backend(LEGACY_PORT),
        host=dict(LEGACY_HOST),
        parameters=dict(state["parameters"]),
        config_sha256=config_sha,
        workload_sha256=workload_sha,
        config=config,
        source={
            "kind": "import",
            "importer": "legacy_main",
            "files": [file_record(path, root)],
            "notes": [
                "run_benchmark.py schema-1 checkpoint; scores and aggregates recomputed from per-case records",
                "IFEval case verdict = official strict prompt-level result",
            ],
        },
        timings={
            "started_at_utc": state.get("started_at_utc"),
            "completed_at_utc": state.get("completed_at_utc"),
            "updated_at_utc": state.get("updated_at_utc"),
            "server_load_seconds": state.get("server_load_seconds"),
            "wall_seconds": wall,
        },
        suites=suites,
        complete=state.get("completed_at_utc") is not None,
    )
    document["extra"] = {"server_log": state.get("server_log")}
    return document
