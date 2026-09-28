"""Full cycle against a fake OpenAI-compatible server:
run -> result.v1 -> validate -> report / compare-hosts (no network, no GPU)."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest
from conftest import ROOT, humaneval_problems, run_python, write_config
from fakeserver import FakeServer

from sparkbench.cli import main
from sparkbench.compare_hosts import compare
from sparkbench.config import load_config
from sparkbench.jsonutil import read_json
from sparkbench.report import report
from sparkbench.runner import RunError, Runner
from sparkbench.schema import semantic_errors, validate
from sparkbench.suites.humaneval import select_problems


def run_full(tmp_path: Path, data_dir: Path, server: FakeServer, label: str) -> dict:
    host = f'\n[host]\nname = "{label}"\ngpu = "NVIDIA GB10"\n'
    config = load_config(write_config(tmp_path, data_dir, server.url, label=label, extra=host))
    path = Runner(config, executor=run_python, log=lambda message: None).run()
    return read_json(path)


@pytest.fixture(scope="module")
def two_hosts(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    from conftest import FIXTURES

    tmp = tmp_path_factory.mktemp("hosts")
    data = tmp / "data"
    import gzip
    import shutil

    shutil.copytree(FIXTURES / "data", data)
    with gzip.open(data / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as handle:
        for problem in humaneval_problems():
            handle.write(json.dumps(problem) + "\n")
    with FakeServer() as first:
        a = run_full(tmp / "a", data, first, "spark-1")
    with FakeServer(speed=0.8, variant="other") as second:
        b = run_full(tmp / "b", data, second, "spark-2")
    return a, b


def test_full_run_produces_valid_v1(two_hosts: tuple[dict, dict]) -> None:
    a, _ = two_hosts
    assert validate(a) == []
    assert semantic_errors(a) == []
    assert a["schema"] == "sparkbench.result.v1"
    assert a["complete"] is True and a["smoke"] is False
    assert a["runtime"]["reported_version"] == "b1-fake"
    assert a["backend"]["extra_body"] == {"cache_prompt": False}
    assert a["host"]["name"] == "spark-1" and a["host"]["gpu"] == "NVIDIA GB10"
    assert list(a["suites"]) == ["mmlu_pro", "gsm8k", "humaneval", "ifeval", "long_context", "tools", "stability_100"]
    sample = [p["task_id"] for p in select_problems(humaneval_problems(), 42)]
    expected_pass = {
        "mmlu_pro": 2,
        "gsm8k": 3,
        "humaneval": 40 - ("HumanEval/7" in sample),
        "ifeval": 2,
        "long_context": 9,
        "tools": 19,
        "stability_100": 100,
    }
    assert {name: suite["verdicts"]["pass"] for name, suite in a["suites"].items()} == expected_pass
    mmlu = a["suites"]["mmlu_pro"]
    assert mmlu["cases"][0]["api_calls"] == 2 and len(mmlu["cases"][0]["transcript"]) == 1
    assert a["overall"]["api_calls"] == a["overall"]["cases"] + len(mmlu["cases"])
    assert a["suites"]["tools"]["cases"][0]["tool_calls"][0]["name"] == "get_weather"
    for suite in a["suites"].values():
        assert suite["timing_source"] == "server"
        for case in suite["cases"]:
            assert case["tok_s"]["decode"] == pytest.approx(30.0)
    assert a["suites"]["long_context"]["cases"][-1]["detail"]["actual_tokens"] <= 60000


def test_report_between_two_runs(two_hosts: tuple[dict, dict]) -> None:
    a, b = two_hosts
    summary, markdown = report(a, b)
    assert summary["speed"]["overall"]["a_vs_b"] == pytest.approx(0.25)
    assert summary["paired"]["gsm8k"]["a_only"] == 1
    assert "| GSM8K-100 | 100.0% | 66.7% | 30.00 | 24.00 | -20.0% |" in markdown
    assert "spark-1 is 25.0% faster than spark-2 (spark-2 as denominator)." in markdown


def test_compare_hosts_lists_numeric_differences(two_hosts: tuple[dict, dict]) -> None:
    a, b = two_hosts
    result = compare(a, b)
    assert result["checks"] == ["config_sha256 differs (host-specific paths/URLs may legitimately differ)"]
    numeric = result["suites"]["gsm8k"]["numeric_differences"]
    assert len(numeric) == 1
    assert numeric[0]["a_final_number"] == "7" and numeric[0]["b_final_number"] == "8"
    assert len(result["suites"]["ifeval"]["text_differences"]) == 1
    assert result["suites"]["tools"]["identical"] == 20
    assert result["suites"]["stability_100"]["decode_b_vs_a"] == pytest.approx(-0.2)
    assert result["suites"]["stability_100"]["median_case_ratio_b_over_a"] == pytest.approx(0.8)
    assert result["overall"]["verdict_flips"] == 1


def test_cli_validate_report_compare(two_hosts: tuple[dict, dict], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = two_hosts
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")
    assert main(["validate", str(pa), str(pb)]) == 0
    assert main(["report", str(pa), str(pb), "--out", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")]) == 0
    assert "## Paired correctness" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert main(["compare-hosts", str(pa), str(pb)]) == 0
    out = capsys.readouterr().out
    assert "## Differences in numbers" in out and "| gsm8k |" in out
    assert main(["compare-hosts", str(pa), str(pb), "--fail-on-diff"]) == 1
    broken = dict(a)
    broken["suites"] = json.loads(json.dumps(a["suites"]))
    broken["suites"]["tools"]["verdicts"]["pass"] = 20
    pa.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["validate", str(pa)]) == 1
    del broken["host"]
    pa.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["validate", str(pa)]) == 1
    assert "missing required property 'host'" in capsys.readouterr().out


def test_cli_smoke_run_and_resume(tmp_path: Path, data_dir: Path) -> None:
    with FakeServer() as server:
        config = write_config(tmp_path, data_dir, server.url, label="smoke")
        args = ["run", "--config", str(config), "--smoke", "--suite", "tools", "--suite", "stability_100", "--suite", "ifeval"]
        assert main(args) == 0
        out = tmp_path / "out" / "smoke_smoke.json"
        document = read_json(out)
        assert validate(document) == []
        assert document["smoke"] is True
        assert [len(s["cases"]) for s in document["suites"].values()] == [2, 3, 2]
        before = len(server.state.requests)
        assert main(args) == 0  # resume: every case is already checkpointed
        chats = [r for r in server.state.requests[before:] if r["path"] == "/v1/chat/completions"]
        assert len(chats) == 1  # only the warm-up request
        assert all(r["payload"]["cache_prompt"] is False for r in chats)
    changed = tmp_path / "changed.toml"
    changed.write_text(config.read_text(encoding="utf-8").replace("mtp_depth = 7", "mtp_depth = 8"), encoding="utf-8")
    with pytest.raises(RunError):
        Runner(load_config(changed), smoke=True).run()


def test_openai_flavor_without_timings(tmp_path: Path, data_dir: Path) -> None:
    """vLLM/SGLang style: no llama.cpp timings, /tokenize {model, prompt}, no extra fields."""
    with FakeServer(flavor="vllm") as server:
        config = load_config(write_config(tmp_path, data_dir, server.url, label="vllm", tokenizer="openai-tokenize", extra_body="{}"))
        path = Runner(config, suites=["long_context", "tools"], log=lambda message: None).run()
        payloads = [r["payload"] for r in server.state.requests]
    document = read_json(path)
    assert validate(document) == [] and semantic_errors(document) == []
    assert document["runtime"]["reported_version"] == "0.0-fake"
    assert all("cache_prompt" not in p for p in payloads)
    assert any(p.get("model") == "fake-model" and "prompt" in p for p in payloads)  # tokenize call
    for suite in document["suites"].values():
        assert suite["timing_source"] == "wall"
        assert suite["aggregate"]["draft_acceptance"] is None
        assert suite["aggregate"]["decode_tokens_per_second"] > 0
    assert document["suites"]["long_context"]["verdicts"]["pass"] == 9


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_managed_server_start_and_stop(tmp_path: Path, data_dir: Path) -> None:
    port = free_port()
    command = json.dumps([sys.executable, str(ROOT / "tests" / "fakeserver.py"), "--port", str(port)])
    extra = f"\n[server]\ncommand = {command}\nready_timeout = 30\nstop_timeout = 10\n"
    config = load_config(write_config(tmp_path, data_dir, f"http://127.0.0.1:{port}", label="managed", extra=extra))
    runner = Runner(config, suites=["stability_100"], smoke=True, log=lambda message: None)
    path = runner.run()
    document = read_json(path)
    assert document["backend"]["managed_server"] is True
    assert document["timings"]["server_load_seconds"] is not None
    assert document["suites"]["stability_100"]["verdicts"]["pass"] == 3
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0  # server stopped
    assert (tmp_path / "out" / "managed_smoke_server.log").exists()
