"""The moved suites behave exactly like the original ``run_benchmark.py``.

Both implementations are driven through the same fake chat function on the
same fixture data; the requests they send (messages, max_tokens, tools) and
the scored case records they produce must be identical.
"""

from __future__ import annotations

import copy
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import IFEVAL_STUB, ROOT, humaneval_problems
from fakeserver import answer, fake_token_count

from sparkbench.suites import REGISTRY
from sparkbench.suites.common import SuiteContext
from sparkbench.suites.humaneval import PASS_MARKER

SUITES = ["mmlu_pro", "gsm8k", "humaneval", "ifeval", "long_context", "tools", "stability_100"]


def load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("legacy_run_benchmark", ROOT / "run_benchmark.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_response(messages: list[dict[str, Any]], max_tokens: int, extra: dict[str, Any]) -> dict[str, Any]:
    content, tool_calls = answer(messages, {"max_tokens": max_tokens, **extra}, "base")
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    predicted = max(1, len(content) // 4)
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"completion_tokens": predicted},
        "timings": {"prompt_n": 10, "prompt_ms": 20.0, "predicted_n": predicted, "predicted_ms": predicted * 25.0, "predicted_per_second": 40.0, "draft_n": 3, "draft_n_accepted": 2},
    }


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], max_tokens: int, **extra: Any) -> tuple[dict[str, Any], float]:
        self.calls.append((model, copy.deepcopy(messages), max_tokens, copy.deepcopy(extra)))
        return fake_response(messages, max_tokens, extra), 0.25

    def token_count(self, text: str) -> int:
        return fake_token_count(text)


class FakeCompleted:
    def __init__(self, program: str) -> None:
        passed = "assert candidate(3) == 3" in program
        self.returncode = 0 if passed else 1
        self.stdout = f"{PASS_MARKER}\n" if passed else "AssertionError\n"


@pytest.fixture
def legacy(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> tuple[ModuleType, Recorder, list[str]]:
    module = load_legacy()
    recorder = Recorder()
    programs: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> FakeCompleted:
        programs.append(kwargs["input"])
        return FakeCompleted(kwargs["input"])

    monkeypatch.setattr(module, "DATA", data_dir)
    monkeypatch.setattr(module, "chat", recorder.chat)
    monkeypatch.setattr(module, "checkpoint", lambda state: None)
    monkeypatch.setattr(module, "load_humaneval", lambda: humaneval_problems())
    monkeypatch.setattr(module, "token_count", recorder.token_count)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.syspath_prepend(str(IFEVAL_STUB))
    return module, recorder, programs


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("name", SUITES)
def test_suite_matches_legacy(name: str, smoke: bool, legacy: tuple[ModuleType, Recorder, list[str]], data_dir: Path) -> None:
    module, old_recorder, old_programs = legacy
    old_state: dict[str, Any] = {"suites": {}}
    module.RUNNERS[name](old_state, "fake-model", smoke)

    new_recorder = Recorder()
    new_programs: list[str] = []

    def executor(program: str, timeout: float) -> tuple[bool, str]:
        assert timeout == 8
        new_programs.append(program)
        completed = FakeCompleted(program)
        return completed.returncode == 0 and PASS_MARKER in completed.stdout, completed.stdout

    suite: dict[str, Any] = {"cases": []}
    ctx = SuiteContext(
        backend=new_recorder,
        model="fake-model",
        smoke=smoke,
        seed=42,
        data_dir=data_dir,
        humaneval_path=data_dir / "HumanEval.jsonl.gz",
        ifeval_lib=IFEVAL_STUB,
        executor=executor,
        log=lambda message: None,
    )
    REGISTRY[name].run(suite, ctx)

    assert new_recorder.calls == old_recorder.calls
    assert len(new_recorder.calls) > 0
    assert new_programs == old_programs
    assert suite == old_state["suites"][name]


def test_docker_command_matches_legacy(legacy: tuple[ModuleType, Recorder, list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = legacy
    seen: list[list[str]] = []

    def capture(command: list[str], **kwargs: Any) -> FakeCompleted:
        seen.append(command)
        return FakeCompleted(kwargs["input"])

    monkeypatch.setattr(module.subprocess, "run", capture)
    problem = humaneval_problems(1)[0]
    module.docker_check(problem, "return x")
    from sparkbench.suites.humaneval import docker_command

    assert seen == [docker_command()]


def test_legacy_request_payload_shape_is_preserved() -> None:
    """OpenAIBackend.chat sends the legacy body when extra_body = {cache_prompt=false}."""
    from sparkbench.backend import OpenAIBackend
    from sparkbench.config import BackendConfig

    sent: dict[str, Any] = {}

    class Capture(OpenAIBackend):
        def post_json(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            sent.update(path=path, payload=payload)
            return {}

    backend = Capture(BackendConfig(base_url="http://x", extra_body={"cache_prompt": False}), 0, 42)
    backend.chat("m", [{"role": "user", "content": "hi"}], 16, tools=[1], tool_choice="auto")
    assert sent["path"] == "/v1/chat/completions"
    assert list(sent["payload"].items()) == [
        ("model", "m"),
        ("messages", [{"role": "user", "content": "hi"}]),
        ("temperature", 0),
        ("seed", 42),
        ("max_tokens", 16),
        ("stream", False),
        ("cache_prompt", False),
        ("tools", [1]),
        ("tool_choice", "auto"),
    ]


def test_legacy_scripts_untouched() -> None:
    """Historical scripts stay byte-identical (the harness copies, never edits)."""
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "run_benchmark.py", "make_report.py", "results/q4.json", "results/q6.json", "launchers", "soak_partial"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if changed.returncode != 0:
        pytest.skip("git not available")
    assert changed.stdout.strip() == ""
