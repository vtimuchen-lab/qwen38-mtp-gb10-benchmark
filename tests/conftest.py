"""Shared fixtures: tiny datasets, a local HumanEval executor, config writer."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
IFEVAL_STUB = FIXTURES / "ifeval_stub"


def humaneval_problems(count: int = 164) -> list[dict[str, str]]:
    """Stub problems; only the count matters for the seed-42 sample of 40."""
    problems = []
    for index in range(count):
        expected = 4 if index == 7 else 3  # problem 7 always fails
        problems.append(
            {
                "task_id": f"HumanEval/{index}",
                "prompt": f'def f{index}(x):\n    """Return x unchanged."""\n',
                "entry_point": f"f{index}",
                "canonical_solution": "    return x\n",
                "test": f"\n\ndef check(candidate):\n    assert candidate(3) == {expected}\n",
            }
        )
    return problems


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(FIXTURES / "data", target)
    with gzip.open(target / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as handle:
        for problem in humaneval_problems():
            handle.write(json.dumps(problem) + "\n")
    return target


def run_python(program: str, timeout: float) -> tuple[bool, str]:
    """Local executor for the canned test programs (Docker is not available in CI)."""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-"], input=program, text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    output = (completed.stdout + completed.stderr)[-4000:]
    return completed.returncode == 0 and "__HUMANEVAL_PASS__" in output, output


@pytest.fixture
def local_executor() -> Callable[[str, float], tuple[bool, str]]:
    return run_python


def write_config(tmp_path: Path, data_dir: Path, url: str, *, label: str = "fake", extra: str = "", tokenizer: str = "llama.cpp", extra_body: str = "{ cache_prompt = false }") -> Path:
    text = f"""
[run]
label = "{label}"
quant = "q4"
output = "{(tmp_path / 'out' / (label + '.json')).as_posix()}"

[model]
name = "fake-model"
sha256 = "{'ab' * 32}"
quantization = "UD-Q4_K_XL"

[runtime]
name = "fake"
version = "1"

[backend]
base_url = "{url}"
tokenizer = "{tokenizer}"
extra_body = {extra_body}
ready_timeout = 10

[request]
temperature = 0
seed = 42

[parameters]
mtp_depth = 7
context = 65536

[data]
dir = "{data_dir.as_posix()}"
humaneval = "{(data_dir / 'HumanEval.jsonl.gz').as_posix()}"
ifeval_lib = "{IFEVAL_STUB.as_posix()}"
{extra}
"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{label}.toml"
    path.write_text(text, encoding="utf-8")
    return path
