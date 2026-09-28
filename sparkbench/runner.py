"""Execute suites against a backend and write ``sparkbench.result.v1``.

Two files are written next to each other:

* ``<output>.json``      - the v1 result (rewritten after every case);
* ``<output>.raw.json``  - full legacy-style checkpoint with complete server
  responses, used to resume an interrupted run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sparkbench.backend import BackendError, ChatBackend, ManagedServer, OpenAIBackend
from sparkbench.config import Config
from sparkbench.hostinfo import collect_host
from sparkbench.jsonutil import read_json, save_json, sha256_file, utc_now
from sparkbench.suites import REGISTRY
from sparkbench.suites.common import Executor, JSONDict, SuiteContext, message_content
from sparkbench.v1 import build_document, suite_from_legacy

RAW_SCHEMA = "sparkbench.raw.v1"


class RunError(RuntimeError):
    """The run cannot start or resume safely."""


def output_paths(config: Config, smoke: bool, output: str | None = None) -> tuple[Path, Path]:
    path = Path(output or config.output)
    if smoke and output is None:
        path = path.with_name(path.stem + "_smoke" + path.suffix)
    return path, path.with_name(path.stem + ".raw.json")


def _model_block(config: Config) -> JSONDict:
    sha = config.model_sha256
    if sha is None and config.model_hash_file and config.model_file:
        sha = sha256_file(Path(config.model_file))
    return {
        "name": config.model_name,
        "file": config.model_file,
        "sha256": sha,
        "quantization": config.model_quantization,
        "source": config.model_source,
    }


def _load_state(raw_path: Path, config: Config, smoke: bool, fresh: bool) -> JSONDict:
    if raw_path.exists() and not fresh:
        state: JSONDict = read_json(raw_path)
        if state.get("config_sha256") != config.config_sha256 or bool(state.get("smoke")) != smoke:
            raise RunError(
                f"{raw_path} was produced by a different config or mode; "
                "use --fresh to start over or --output to write elsewhere"
            )
        return state
    return {
        "schema": RAW_SCHEMA,
        "config_sha256": config.config_sha256,
        "smoke": smoke,
        "started_at_utc": utc_now(),
        "suites": {},
    }


class Runner:
    def __init__(
        self,
        config: Config,
        *,
        suites: list[str] | None = None,
        smoke: bool = False,
        output: str | None = None,
        fresh: bool = False,
        backend: ChatBackend | None = None,
        executor: Executor | None = None,
        suite_options: dict[str, JSONDict] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.suites = list(suites or config.suites)
        unknown = [name for name in self.suites if name not in REGISTRY]
        if unknown:
            raise RunError(f"unknown suite(s): {', '.join(unknown)}")
        self.smoke = smoke
        self.output, self.raw_path = output_paths(config, smoke, output)
        self.fresh = fresh
        self.openai = OpenAIBackend(config.backend, config.temperature, config.seed, config.model_name)
        self.backend: ChatBackend = backend or self.openai
        self.executor = executor
        self.suite_options = suite_options or {}
        self.log = log or (lambda message: print(message, flush=True))
        self.state: JSONDict = {}
        self.host: JSONDict = {}
        self.runtime: JSONDict = dict(config.runtime)
        self.model = _model_block(config)

    # -- persistence ----------------------------------------------------
    def checkpoint(self) -> None:
        self.state["updated_at_utc"] = utc_now()
        save_json(self.raw_path, self.state)
        save_json(self.output, self.document())

    def document(self) -> JSONDict:
        config = self.config
        done = self.state.get("completed_suites", [])
        suites = {
            name: suite_from_legacy(name, self.state["suites"][name], complete=name in done)
            for name in self.suites
            if name in self.state["suites"] and self.state["suites"][name].get("cases")
        }
        parameters = {**config.parameters, "temperature": config.temperature, "seed": config.seed}
        return build_document(
            label=config.label,
            quant=config.quant,
            model=self.model,
            runtime=self.runtime,
            backend={
                "kind": "openai",
                "base_url": config.backend.base_url,
                "api": "/v1/chat/completions",
                "managed_server": config.server is not None,
                "tokenizer": config.backend.tokenizer,
                "extra_body": config.backend.extra_body,
            },
            host=self.host,
            parameters=parameters,
            config_sha256=config.config_sha256,
            workload_sha256=config.workload_sha256,
            config=config.raw,
            source={"kind": "run", "notes": [f"config file: {config.path}"]},
            timings={
                "started_at_utc": self.state.get("started_at_utc"),
                "completed_at_utc": self.state.get("completed_at_utc"),
                "updated_at_utc": self.state.get("updated_at_utc"),
                "server_load_seconds": self.state.get("server_load_seconds"),
                "wall_seconds": sum(s["aggregate"]["wall_seconds"] for s in suites.values()),
            },
            suites=suites,
            smoke=self.smoke,
            complete=all(name in done for name in self.suites),
        )

    # -- execution ------------------------------------------------------
    def warmup(self) -> None:
        response, _ = self.backend.chat(self.config.model_name, [{"role": "user", "content": "Reply with OK."}], 16)
        if not message_content(response).strip():
            raise BackendError("empty warm-up response")

    def context(self, name: str) -> SuiteContext:
        config = self.config
        options: dict[str, Any] = {**config.suite_options.get(name, {}), **self.suite_options.get(name, {})}
        return SuiteContext(
            backend=self.backend,
            model=config.model_name,
            smoke=self.smoke,
            seed=config.seed,
            data_dir=Path(config.data.dir),
            humaneval_path=Path(config.data.humaneval),
            ifeval_lib=Path(config.data.ifeval_lib),
            options=options,
            checkpoint=self.checkpoint,
            log=self.log,
            executor=self.executor,
        )

    def run(self) -> Path:
        config = self.config
        if config.data.nltk_data:
            os.environ.setdefault("NLTK_DATA", config.data.nltk_data)
        self.state = _load_state(self.raw_path, config, self.smoke, self.fresh)
        self.state.pop("completed_at_utc", None)
        self.host = collect_host(config.host)
        server: ManagedServer | None = None
        try:
            if config.server is not None:
                server = ManagedServer(config.server, self.openai, self.output.with_name(self.output.stem + "_server.log"))
                load_seconds = server.start()
                self.state["server_load_seconds"] = load_seconds
                self.log(f"{config.label}: server ready in {load_seconds:.2f}s")
            elif self.backend is self.openai:
                self.openai.wait_ready(config.backend.ready_timeout)
            if self.backend is self.openai:
                self.runtime.update(self.openai.probe_runtime())
            self.checkpoint()
            if config.warmup:
                self.warmup()
            completed: list[str] = self.state.setdefault("completed_suites", [])
            for name in self.suites:
                self.log(f"=== {config.label} {name} ===")
                if name in completed:
                    completed.remove(name)
                suite = self.state["suites"].setdefault(name, {"cases": []})
                started = time.monotonic()
                REGISTRY[name].run(suite, self.context(name))
                suite["suite_wall_seconds"] = suite.get("suite_wall_seconds", 0) + time.monotonic() - started
                completed.append(name)
                self.checkpoint()
            self.state["completed_at_utc"] = utc_now()
            self.checkpoint()
        finally:
            if server is not None:
                server.stop()
        return self.output
