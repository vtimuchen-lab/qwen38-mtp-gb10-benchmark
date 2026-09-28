"""Shared helpers for converting historical result files into result.v1.

Importers only read the source files; they never modify them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sparkbench.config import workload_view
from sparkbench.jsonutil import sha256_file, sha256_json
from sparkbench.suites.common import JSONDict

# Facts about the machine that produced every historical file, taken from the
# README "Test system" table. The historical JSON did not record a hostname.
LEGACY_HOST: JSONDict = {
    "name": None,
    "gpu": "NVIDIA GB10",
    "driver": "580.173.02",
    "arch": "aarch64",
    "kernel": "6.17.0-1029-nvidia",
    "os": None,
    "cpus": 20,
    "source": "README.md 'Test system' table (legacy import; hostname was not recorded)",
}

# llama.cpp commit reported in README for every historical run.
LEGACY_RUNTIME_VERSION = "d2f83055d6e3b379b5d34c4837122a918cf402c2"

QUANTIZATION = {"q4": "UD-Q4_K_XL", "q6": "UD-Q6_K_XL"}


def legacy_runtime(binary: str | None) -> JSONDict:
    return {"name": "llama.cpp", "version": LEGACY_RUNTIME_VERSION, "binary": binary}


def legacy_backend(port: int, extra_body: JSONDict | None = None) -> JSONDict:
    return {
        "kind": "openai",
        "base_url": f"http://127.0.0.1:{port}",
        "api": "/v1/chat/completions",
        "managed_server": True,
        "tokenizer": "llama.cpp",
        "extra_body": extra_body if extra_body is not None else {"cache_prompt": False},
    }


def llama_server_flags(ctx_size: int, depth: int | None, port: int) -> list[str]:
    """The llama-server flags shared by every historical series."""
    flags = [
        "--host", "127.0.0.1", "--port", str(port), "--no-webui", "--offline",
        "--ctx-size", str(ctx_size), "--parallel", "1", "--gpu-layers", "all",
        "--fit", "off", "--flash-attn", "on", "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0", "--batch-size", "2048", "--ubatch-size", "2048",
        "--jinja", "--reasoning", "off", "--reasoning-budget", "0",
        "--reasoning-format", "deepseek",
    ]  # fmt: skip
    if depth is not None:
        flags += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(depth), "--spec-draft-ngl", "all"]
    return [*flags, "--log-verbosity", "2"]


def file_record(path: Path, root: Path) -> JSONDict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def config_hashes(config: JSONDict) -> tuple[str, str]:
    """(config_sha256, workload_sha256) for a reconstructed legacy config."""
    return sha256_json(config), sha256_json(workload_view(config))


def pseudo_response(
    *,
    content: str | None,
    timings: JSONDict | None,
    wall_seconds: float | None,
    finish_reason: str | None = None,
    usage: JSONDict | None = None,
    message: JSONDict | None = None,
) -> JSONDict:
    """Shape a historical record like ``compact_response`` output."""
    return {
        "content": content,
        "message": message or {},
        "finish_reason": finish_reason,
        "usage": usage or {},
        "timings": timings or {},
        "wall_seconds": wall_seconds,
    }


def model_block(name: str, file: str | None, sha256: str | None, quantization: str | None) -> JSONDict:
    return {
        "name": name,
        "file": file,
        "sha256": sha256,
        "quantization": quantization,
        "source": "unsloth/Qwen3.8-27B-GGUF",
    }


def sha_from_blob_path(path: str | None) -> str | None:
    if not path:
        return None
    name = Path(path).name
    return name.removeprefix("sha256-") if name.startswith("sha256-") else None


def int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)
