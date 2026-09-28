"""TOML run configuration.

Every machine-specific value (server binary, model path, URL, data paths)
lives in a TOML file; nothing host-specific is hard-coded in the package.
Relative paths are resolved against the current working directory, so run
the CLI from the repository root.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparkbench.jsonutil import sha256_json

ALL_SUITES = (
    "mmlu_pro",
    "gsm8k",
    "humaneval",
    "ifeval",
    "long_context",
    "tools",
    "stability_100",
)

TOKENIZERS = ("llama.cpp", "openai-tokenize", "none")

_SECTIONS: dict[str, set[str]] = {
    "run": {"label", "quant", "suites", "output", "warmup"},
    "model": {"name", "file", "sha256", "hash_file", "quantization", "source"},
    "runtime": {"name", "version", "binary"},
    "backend": {
        "kind",
        "base_url",
        "api_key_env",
        "timeout",
        "health_path",
        "tokenizer",
        "extra_body",
        "ready_timeout",
    },
    "server": {"command", "env", "cwd", "log", "ready_timeout", "stop_signal", "stop_timeout"},
    "request": {"temperature", "seed"},
    "host": {"name", "gpu", "driver", "labels"},
    "parameters": set(),  # free-form, recorded verbatim
    "data": {"dir", "humaneval", "ifeval_lib", "nltk_data"},
    "suites": set(),  # per-suite option tables, validated below
}

_SUITE_OPTIONS: dict[str, set[str]] = {
    "humaneval": {"executor", "docker_image", "timeout"},
}


class ConfigError(ValueError):
    """Raised for malformed or unknown configuration keys."""


@dataclass(frozen=True)
class BackendConfig:
    base_url: str
    kind: str = "openai"
    api_key_env: str | None = None
    timeout: float = 1800.0
    health_path: str = "/health"
    tokenizer: str = "llama.cpp"
    extra_body: dict[str, Any] = field(default_factory=dict)
    ready_timeout: float = 600.0


@dataclass(frozen=True)
class ServerConfig:
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    log: str | None = None
    ready_timeout: float = 600.0
    stop_signal: str = "SIGINT"
    stop_timeout: float = 45.0


@dataclass(frozen=True)
class DataConfig:
    dir: str = "data"
    humaneval: str = "vendor/human-eval/data/HumanEval.jsonl.gz"
    ifeval_lib: str = "vendor/google-research"
    nltk_data: str | None = None


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: str
    label: str
    quant: str | None
    suites: tuple[str, ...]
    output: str
    warmup: bool
    model_name: str
    model_file: str | None
    model_sha256: str | None
    model_hash_file: bool
    model_quantization: str | None
    model_source: str | None
    runtime: dict[str, Any]
    backend: BackendConfig
    server: ServerConfig | None
    temperature: int | float
    seed: int
    host: dict[str, Any]
    parameters: dict[str, Any]
    data: DataConfig
    suite_options: dict[str, dict[str, Any]]

    @property
    def config_sha256(self) -> str:
        """Hash of the whole parsed config (formatting/comments do not matter)."""
        return sha256_json(self.raw)

    @property
    def workload_sha256(self) -> str:
        return sha256_json(workload_view(self.raw))


def workload_view(raw: dict[str, Any]) -> dict[str, Any]:
    """The portable part of a config: what is measured, not where.

    Drops machine-specific sections (run, host, server, data, backend URL) and
    file paths, so the same workload on two hosts hashes identically.
    """
    portable: dict[str, Any] = {
        key: value for key, value in raw.items() if key not in {"run", "host", "server", "data", "backend"}
    }
    if isinstance(portable.get("model"), dict):
        portable["model"] = {k: v for k, v in portable["model"].items() if k not in {"file", "hash_file"}}
    if isinstance(portable.get("runtime"), dict):
        portable["runtime"] = {k: v for k, v in portable["runtime"].items() if k != "binary"}
    if isinstance(portable.get("suites"), dict):
        portable["suites"] = {
            name: {k: v for k, v in options.items() if k != "executor"}
            for name, options in portable["suites"].items()
            if isinstance(options, dict)
        }
    backend = raw.get("backend", {})
    portable["backend"] = {
        "extra_body": backend.get("extra_body", {}),
        "tokenizer": backend.get("tokenizer", "llama.cpp"),
    }
    return portable


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    allowed = _SECTIONS[name]
    if allowed:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")
    return value


def _str_or_none(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _number(value: Any, key: str) -> int | float:
    # Keep ints as ints so the request body matches the legacy runner byte-for-byte.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return value


def parse_config(raw: dict[str, Any], path: str = "<memory>") -> Config:
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    run = _table(raw, "run")
    model = _table(raw, "model")
    runtime = _table(raw, "runtime")
    backend = _table(raw, "backend")
    server = _table(raw, "server")
    request = _table(raw, "request")
    host = _table(raw, "host")
    parameters = _table(raw, "parameters")
    data = _table(raw, "data")
    suites_table = _table(raw, "suites")

    suites = tuple(run.get("suites", list(ALL_SUITES)))
    for suite in suites:
        if suite not in ALL_SUITES:
            raise ConfigError(f"unknown suite {suite!r}; expected one of {', '.join(ALL_SUITES)}")
    suite_options: dict[str, dict[str, Any]] = {}
    for name, options in suites_table.items():
        if name not in ALL_SUITES or not isinstance(options, dict):
            raise ConfigError(f"[suites.{name}] is not a known suite option table")
        extra = sorted(set(options) - _SUITE_OPTIONS.get(name, set()))
        if extra:
            raise ConfigError(f"unknown key(s) in [suites.{name}]: {', '.join(extra)}")
        suite_options[name] = dict(options)

    if "base_url" not in backend:
        raise ConfigError("[backend] base_url is required")
    if backend.get("kind", "openai") != "openai":
        raise ConfigError("[backend] kind must be 'openai' (any OpenAI-compatible server)")
    tokenizer = backend.get("tokenizer", "llama.cpp")
    if tokenizer not in TOKENIZERS:
        raise ConfigError(f"[backend] tokenizer must be one of {', '.join(TOKENIZERS)}")
    extra_body = backend.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ConfigError("[backend] extra_body must be a table")
    if "name" not in model:
        raise ConfigError("[model] name (the model id sent to the server) is required")

    server_config: ServerConfig | None = None
    if server:
        command = server.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ConfigError("[server] command must be a non-empty list of strings")
        server_config = ServerConfig(
            command=list(command),
            env={str(k): str(v) for k, v in dict(server.get("env", {})).items()},
            cwd=_str_or_none(server.get("cwd"), "server.cwd"),
            log=_str_or_none(server.get("log"), "server.log"),
            ready_timeout=float(server.get("ready_timeout", backend.get("ready_timeout", 600))),
            stop_signal=str(server.get("stop_signal", "SIGINT")),
            stop_timeout=float(server.get("stop_timeout", 45)),
        )

    label = str(run.get("label") or run.get("quant") or (Path(path).stem if path != "<memory>" else "run"))
    return Config(
        raw=raw,
        path=path,
        label=label,
        quant=_str_or_none(run.get("quant"), "run.quant"),
        suites=suites,
        output=str(run.get("output", f"results/runs/{label}.json")),
        warmup=bool(run.get("warmup", True)),
        model_name=str(model["name"]),
        model_file=_str_or_none(model.get("file"), "model.file"),
        model_sha256=_str_or_none(model.get("sha256"), "model.sha256"),
        model_hash_file=bool(model.get("hash_file", False)),
        model_quantization=_str_or_none(model.get("quantization"), "model.quantization"),
        model_source=_str_or_none(model.get("source"), "model.source"),
        runtime=dict(runtime),
        backend=BackendConfig(
            base_url=str(backend["base_url"]).rstrip("/"),
            api_key_env=_str_or_none(backend.get("api_key_env"), "backend.api_key_env"),
            timeout=float(backend.get("timeout", 1800)),
            health_path=str(backend.get("health_path", "/health")),
            tokenizer=tokenizer,
            extra_body=dict(extra_body),
            ready_timeout=float(backend.get("ready_timeout", 600)),
        ),
        server=server_config,
        temperature=_number(request.get("temperature", 0), "request.temperature"),
        seed=int(request.get("seed", 42)),
        host=dict(host),
        parameters=dict(parameters),
        data=DataConfig(
            dir=str(data.get("dir", "data")),
            humaneval=str(data.get("humaneval", DataConfig.humaneval)),
            ifeval_lib=str(data.get("ifeval_lib", DataConfig.ifeval_lib)),
            nltk_data=_str_or_none(data.get("nltk_data"), "data.nltk_data"),
        ),
        suite_options=suite_options,
    )


def load_config(path: str | Path) -> Config:
    file = Path(path)
    with file.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_config(raw, str(file))
