"""OpenAI-compatible chat backend and optional managed server process.

Works with any server that exposes ``/v1/chat/completions`` (llama.cpp
``llama-server``, SGLang, vLLM, ...). Server-specific extras are opt-in via
config: ``extra_body`` is merged into every request, ``tokenizer`` selects the
token-count endpoint used by the long-context suite.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Any, Protocol

from sparkbench.config import BackendConfig, ServerConfig


class BackendError(RuntimeError):
    """HTTP or protocol failure while talking to the model server."""


class ServerStartError(BackendError):
    """A managed server could not be started or never became healthy."""


class ChatBackend(Protocol):
    """What suites need from a backend; tests substitute their own."""

    def chat(
        self, model: str, messages: list[dict[str, Any]], max_tokens: int, **extra: Any
    ) -> tuple[dict[str, Any], float]: ...

    def token_count(self, text: str) -> int: ...


class OpenAIBackend:
    def __init__(
        self, config: BackendConfig, temperature: int | float = 0, seed: int = 42, model: str = ""
    ) -> None:
        self.config = config
        self.model = model
        self.temperature = temperature
        self.seed = seed

    # -- transport -----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def post_json(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        url = self.config.base_url + path
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.config.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"cannot reach {url}: {exc.reason}") from exc
        if not isinstance(value, dict):
            raise BackendError(f"unexpected non-object JSON from {url}")
        return value

    def get_json(self, path: str, timeout: float = 5.0) -> dict[str, Any] | None:
        request = urllib.request.Request(self.config.base_url + path, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None
        return value if isinstance(value, dict) else None

    # -- API -----------------------------------------------------------
    def chat(
        self, model: str, messages: list[dict[str, Any]], max_tokens: int, **extra: Any
    ) -> tuple[dict[str, Any], float]:
        # Key order and values mirror the legacy run_benchmark.chat() payload;
        # llama.cpp's ``cache_prompt: false`` now comes from extra_body.
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(self.config.extra_body)
        payload.update(extra)
        started = time.monotonic()
        response = self.post_json("/v1/chat/completions", payload)
        return response, time.monotonic() - started

    def token_count(self, text: str) -> int:
        tokenizer = self.config.tokenizer
        if tokenizer == "llama.cpp":
            response = self.post_json("/tokenize", {"content": text, "add_special": False}, timeout=180)
            return len(response["tokens"])
        if tokenizer == "openai-tokenize":
            # vLLM and SGLang style: POST /tokenize {model, prompt} -> {tokens|count}
            response = self.post_json(
                "/tokenize",
                {"model": self.model, "prompt": text, "add_special_tokens": False},
                timeout=180,
            )
            if "count" in response:
                return int(response["count"])
            return len(response["tokens"])
        raise BackendError("backend.tokenizer = 'none': this suite needs a token-count endpoint")

    def healthy(self) -> bool:
        request = urllib.request.Request(self.config.base_url + self.config.health_path)
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return bool(response.status == 200)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def wait_ready(self, timeout: float, process: subprocess.Popen[str] | None = None, log_path: Path | None = None) -> float:
        started = time.monotonic()
        deadline = started + timeout
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                tail = ""
                if log_path is not None and log_path.exists():
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
                raise BackendError(f"server exited with {process.returncode}:\n{tail}")
            if self.healthy():
                return time.monotonic() - started
            time.sleep(0.5)
        raise BackendError(f"server did not become ready within {timeout:.0f} seconds")

    def probe_runtime(self) -> dict[str, Any]:
        """Best-effort runtime identification; never fails the run."""
        found: dict[str, Any] = {}
        props = self.get_json("/props")  # llama.cpp
        if props and "build_info" in props:
            found["reported_version"] = str(props["build_info"])
            found["reported_by"] = "/props"
            return found
        version = self.get_json("/version")  # vLLM
        if version and "version" in version:
            found["reported_version"] = str(version["version"])
            found["reported_by"] = "/version"
            return found
        info = self.get_json("/get_server_info")  # SGLang
        if info and "version" in info:
            found["reported_version"] = str(info["version"])
            found["reported_by"] = "/get_server_info"
        return found


class ManagedServer:
    """Start/stop a model server from a config-supplied command."""

    def __init__(self, config: ServerConfig, backend: OpenAIBackend, default_log: Path) -> None:
        self.config = config
        self.backend = backend
        self.log_path = Path(config.log) if config.log else default_log
        self.process: subprocess.Popen[str] | None = None
        self._log: IO[str] | None = None

    def start(self) -> float:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env.update(self.config.env)
        try:
            self.process = subprocess.Popen(
                self.config.command,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.config.cwd,
                env=env,
                # Own process group, so stop() also reaches worker processes
                # (SGLang/vLLM spawn schedulers that would otherwise keep the port).
                start_new_session=True,
            )
            return self.backend.wait_ready(self.config.ready_timeout, self.process, self.log_path)
        except OSError as exc:
            raise ServerStartError(f"cannot start {self.config.command[0]!r}: {exc}") from exc
        except BackendError as exc:
            raise ServerStartError(str(exc)) from exc

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            self._signal(process, getattr(signal, self.config.stop_signal, signal.SIGINT))
            try:
                process.wait(timeout=self.config.stop_timeout)
            except subprocess.TimeoutExpired:
                self._signal(process, signal.SIGKILL)
                process.wait(timeout=15)
        if process is not None:
            # The leader may be gone while workers still hold GPU memory or the port.
            self._signal(process, signal.SIGKILL)
        if self._log is not None:
            self._log.close()
            self._log = None

    @staticmethod
    def _signal(process: subprocess.Popen[str], sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
