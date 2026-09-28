"""``sparkbench sweep``: grid expansion, managed fake servers per point,
failed points, resume and the Pareto/selection report (no network, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import ROOT, write_config
from test_e2e_fake_server import free_port

from sparkbench.cli import main
from sparkbench.jsonutil import read_json, save_json
from sparkbench.schema import validate
from sparkbench.sweep import (
    PointRow,
    SelectRule,
    SweepError,
    build_report,
    describe_plan,
    expand_points,
    load_sweep,
    pareto_front,
    point_config,
    run_sweep,
    select_point,
    substitute,
)

FAKESERVER = (ROOT / "tests" / "fakeserver.py").as_posix()

GRID_2X2 = """
[grid]
mode = [
  { label = "exact", speed = 1.0, variant = "base", extra = [] },
  { label = "fast", speed = 2.0, variant = "sloppy", extra = [] },
]
mem = [0.8, 0.9]
"""

SELECT = """
[select]
rule = "max_speed_within_baseline"
baseline = { mode = "exact", mem = 0.8 }
max_quality_drop_pp = 1.0
"""


def make_sweep(tmp_path: Path, data_dir: Path, grid: str = GRID_2X2, select: str = SELECT, sweep_extra: str = "") -> Path:
    port = free_port()
    command = json.dumps(
        [sys.executable, FAKESERVER, "--port", str(port), "--speed", "{speed}", "--scale", "{mem}", "--variant", "{variant}", "{extra}"]
    )
    server = f"\n[server]\ncommand = {command}\nready_timeout = 30\nstop_timeout = 10\n"
    base = write_config(tmp_path, data_dir, f"http://127.0.0.1:{port}", label="base", extra=server)
    text = f"""
[sweep]
name = "fake-sweep"
base_config = "{base.name}"
output_dir = "{(tmp_path / 'sweep').as_posix()}"
suites = ["tools", "stability_100"]
smoke = true
{sweep_extra}
{grid}
{select}
"""
    path = tmp_path / "sweep.toml"
    path.write_text(text, encoding="utf-8")
    return path


def quiet(message: str) -> None:
    return None


def statuses(outcomes: list) -> dict[str, str]:
    return {outcome.point.id: outcome.status for outcome in outcomes}


def test_grid_2x2_on_fake_servers(tmp_path: Path, data_dir: Path) -> None:
    sweep = load_sweep(make_sweep(tmp_path, data_dir))
    outcomes = run_sweep(sweep, log=quiet)
    ids = ["mode-exact__mem-0.8", "mode-exact__mem-0.9", "mode-fast__mem-0.8", "mode-fast__mem-0.9"]
    assert statuses(outcomes) == dict.fromkeys(ids, "done")

    documents = {point_id: read_json(tmp_path / "sweep" / "points" / f"{point_id}.json") for point_id in ids}
    for point_id, document in documents.items():
        assert validate(document) == []
        assert document["schema"] == "sparkbench.result.v1" and document["complete"] and document["smoke"]
        assert document["label"] == f"fake-sweep-{point_id}"
        assert list(document["suites"]) == ["tools", "stability_100"]
        assert document["backend"]["managed_server"] is True
        assert (tmp_path / "sweep" / "points" / f"{point_id}.server.log").exists()
    # substituted values reach the server (decode rate = 30 * speed * mem) and the result
    speeds = {point_id: doc["overall"]["decode_tokens_per_second"] for point_id, doc in documents.items()}
    assert speeds == pytest.approx({ids[0]: 24.0, ids[1]: 27.0, ids[2]: 48.0, ids[3]: 54.0})
    assert documents[ids[3]]["parameters"]["sweep_point"]["mode"]["label"] == "fast"
    assert len({doc["workload_sha256"] for doc in documents.values()}) == 4
    assert documents[ids[3]]["suites"]["stability_100"]["verdicts"]["pass"] == 0  # "sloppy" costs quality

    report = build_report(sweep)
    assert report["counts"] == {"done": 4, "failed": 0, "pending": 0}
    assert report["pareto_front"] == [ids[3], ids[1]]
    assert report["selection"]["reference"] == ids[0]
    assert report["selection"]["chosen"] == ids[1]  # fastest within 1 pp of the baseline
    assert main(["sweep", "--config", sweep.path, "--report-only"]) == 0
    markdown = (tmp_path / "sweep" / "report.md").read_text(encoding="utf-8")
    assert "**Recommended: `mode-exact__mem-0.9`** (mode=exact, mem=0.9)" in markdown
    assert "x vs reference" in markdown and "| mode-exact__mem-0.8 | exact | 0.8 | done |" in markdown
    assert "| mode-fast__mem-0.9 | fast | 0.9 | done |" in markdown and "## Pareto front" in markdown
    assert read_json(tmp_path / "sweep" / "report.json")["schema"] == "sparkbench.sweep_report.v1"


def test_failed_point_is_recorded_and_sweep_continues(tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    grid = """
[grid]
mode = [
  { label = "oom", speed = 1.0, variant = "base", extra = ["--crash"] },
  { label = "hang", speed = 1.0, variant = "base", extra = ["--hang"] },
  { label = "exact", speed = 1.0, variant = "base", extra = [] },
]
mem = [1.0]
"""
    path = make_sweep(tmp_path, data_dir, grid=grid, select="", sweep_extra="ready_timeout = 2")
    assert main(["sweep", "--config", str(path)]) == 1
    points = tmp_path / "sweep" / "points"
    oom = read_json(points / "mode-oom__mem-1.0.failed.json")
    assert oom["schema"] == "sparkbench.sweep_failure.v1" and oom["stage"] == "server_start"
    assert "exited with 3" in oom["error"]
    assert "CUDA out of memory" in oom["server_log_tail"]
    assert "--crash" in oom["server_command"]
    hang = read_json(points / "mode-hang__mem-1.0.failed.json")
    assert hang["stage"] == "server_start" and "did not become ready within 2 seconds" in hang["error"]
    assert read_json(points / "mode-exact__mem-1.0.json")["complete"] is True

    report = read_json(tmp_path / "sweep" / "report.json")
    assert report["counts"] == {"done": 1, "failed": 2, "pending": 0}
    assert report["selection"]["chosen"] == "mode-exact__mem-1.0"
    markdown = (tmp_path / "sweep" / "report.md").read_text(encoding="utf-8")
    assert "## Failed points" in markdown and "| mode-oom__mem-1.0 | server_start | ServerStartError: server exited with 3 — log: fake: CUDA out of memory" in markdown
    assert "wrote" in capsys.readouterr().out

    # a failed point is retried on the next run; finished points are not
    outcomes = run_sweep(load_sweep(path), log=quiet)
    assert statuses(outcomes) == {"mode-oom__mem-1.0": "failed", "mode-hang__mem-1.0": "failed", "mode-exact__mem-1.0": "cached"}


def test_resume_skips_points_with_the_same_workload(tmp_path: Path, data_dir: Path) -> None:
    path = make_sweep(tmp_path, data_dir)
    sweep = load_sweep(path)
    assert set(statuses(run_sweep(sweep, log=quiet)).values()) == {"done"}
    points = tmp_path / "sweep" / "points"
    stamp = {p.name: p.stat().st_mtime_ns for p in points.glob("*.json")}

    assert set(statuses(run_sweep(sweep, log=quiet)).values()) == {"cached"}
    assert {p.name: p.stat().st_mtime_ns for p in points.glob("*.json")} == stamp  # nothing rewritten

    (points / "mode-fast__mem-0.8.json").unlink()
    tampered = read_json(points / "mode-exact__mem-0.9.json")
    tampered["workload_sha256"] = "0" * 64
    save_json(points / "mode-exact__mem-0.9.json", tampered)
    assert statuses(run_sweep(sweep, log=quiet)) == {
        "mode-exact__mem-0.8": "cached",
        "mode-exact__mem-0.9": "done",
        "mode-fast__mem-0.8": "done",
        "mode-fast__mem-0.9": "cached",
    }
    assert "mode-fast__mem-0.8: skip (complete)" in describe_plan(sweep)

    # a changed workload (base [parameters]) invalidates every finished point
    base = tmp_path / "base.toml"
    base.write_text(base.read_text(encoding="utf-8").replace("mtp_depth = 7", "mtp_depth = 8"), encoding="utf-8")
    plan = describe_plan(load_sweep(path))
    assert plan.count(": run\n") == 4 and "skip" not in plan
    assert build_report(load_sweep(path))["counts"] == {"done": 0, "failed": 0, "pending": 4}


def row(point_id: str, quality: float | None, tps: float | None, **values: object) -> PointRow:
    return PointRow(point_id, dict(values), "done", quality=quality, decode_tps=tps)


def test_pareto_front() -> None:
    rows = [
        row("a", 0.90, 10.0),
        row("b", 0.85, 20.0),
        row("c", 0.80, 15.0),  # dominated by b
        row("d", 0.90, 9.0),  # dominated by a (same quality, slower)
        row("e", 0.70, 30.0),
        row("f", 0.90, 10.0),  # exact tie with a: neither dominates
        row("g", None, 99.0),  # unscored: never on the front
        PointRow("h", {}, "failed"),
    ]
    assert pareto_front(rows) == ["e", "b", "a", "f"]
    assert pareto_front([]) == []


def test_selection_rules() -> None:
    rows = [
        row("base", 0.80, 10.0, algo="off"),
        row("mtp", 0.795, 25.0, algo="mtp"),
        row("eagle", 0.78, 30.0, algo="eagle"),
        row("dflash", 0.82, 20.0, algo="dflash"),
    ]
    within = SelectRule(baseline={"algo": "off"}, max_quality_drop_pp=1.0)
    result = select_point(rows, within)
    assert result["chosen"] == "mtp" and result["reference"] == "base"
    assert result["threshold"] == pytest.approx(0.79)
    assert select_point(rows, SelectRule(baseline={"algo": "off"}, max_quality_drop_pp=2.0))["chosen"] == "eagle"
    assert select_point(rows, SelectRule())["chosen"] == "dflash"  # "best" reference, 0 pp
    assert select_point(rows, SelectRule(rule="max_quality"))["chosen"] == "dflash"
    assert select_point(rows, SelectRule(rule="max_speed_min_quality", min_quality_pct=79.0))["chosen"] == "mtp"
    assert select_point(rows, SelectRule(rule="max_speed_min_quality", min_quality_pct=95.0))["chosen"] is None
    failed = [PointRow("base", {"algo": "off"}, "failed"), *rows[1:]]
    missing = select_point(failed, within)
    assert missing["chosen"] is None and "no measurement" in missing["reason"]


def test_substitution() -> None:
    variables = {"n": 4, "thinking": False, "algo": "EAGLE", "args": ["--steps", "{n}"], "frac": 0.85}
    base = {
        "server": {"command": ["serve", "--algo", "{algo}", "{args}", "--mem={frac}", "--json={{\"x\": {n}}}"]},
        "backend": {"extra_body": {"chat_template_kwargs": {"enable_thinking": "{thinking}"}}},
        "parameters": {"draft": "{n}", "label": "{algo}-{n}-{thinking}"},
    }
    assert substitute(base, variables) == {
        "server": {"command": ["serve", "--algo", "EAGLE", "--steps", 4, "--mem=0.85", '--json={"x": 4}']},  # stringified by point_config
        "backend": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "parameters": {"draft": 4, "label": "EAGLE-4-false"},
    }
    with pytest.raises(SweepError, match="unknown placeholder"):
        substitute(["--x", "{typo}"], variables)
    with pytest.raises(SweepError, match="whole string"):
        substitute("--a {args}", variables)
    with pytest.raises(SweepError, match="nest too deeply"):
        substitute("{loop}", {"loop": "{loop}"})


def test_sweep_config_validation(tmp_path: Path, data_dir: Path) -> None:
    path = make_sweep(tmp_path, data_dir)
    text = path.read_text(encoding="utf-8")

    def broken(old: str, new: str) -> str:
        bad = tmp_path / "bad.toml"
        bad.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(SweepError) as info:
            expand_points(load_sweep(bad))
        return str(info.value)

    assert "unknown key" in broken("smoke = true", "smoke = true\nbogus = 1")
    assert "non-empty subset" in broken('"stability_100"]', '"nope"]')
    assert "label" in broken('label = "fast", ', "")
    assert "rule must be" in broken("max_speed_within_baseline", "fastest")
    assert "unknown: algo" in broken('baseline = { mode = "exact", mem = 0.8 }', 'baseline = { algo = "off" }')
    assert "no points" in broken("[select]", '[[exclude]]\nmode = ["exact", "fast"]\n\n[select]')

    excluded = tmp_path / "excluded.toml"
    extra = '[[exclude]]\nmode = "fast"\nmem = 0.8\n\n[[include]]\nmode = { label = "off", speed = 0.5, variant = "base", extra = [] }\nmem = 1.0\n\n[select]'
    excluded.write_text(text.replace("[select]", extra), encoding="utf-8")
    sweep = load_sweep(excluded)
    points = expand_points(sweep)
    assert [p.id for p in points] == ["mode-exact__mem-0.8", "mode-exact__mem-0.9", "mode-fast__mem-0.9", "mode-off__mem-1.0"]
    config = point_config(sweep, points[-1])
    assert config.server is not None and config.server.command[-6:] == ["--speed", "0.5", "--scale", "1.0", "--variant", "base"]
    assert config.suites == ("tools", "stability_100") and config.output.endswith("points/mode-off__mem-1.0.json")


def test_example_sweep_expands() -> None:
    sweep = load_sweep(ROOT / "examples" / "sweep-qwen38-sglang.toml")
    points = expand_points(sweep)
    assert len(points) == len({p.id for p in points}) >= 8
    commands = {p.id: point_config(sweep, p) for p in points}
    for point_id, config in commands.items():
        assert config.server is not None
        assert not any("{" in arg for arg in config.server.command), point_id
    assert len(points) == 2 + 4 * 2 * 2  # "off" once per thinking mode, others per draft size
    baseline = [p for p in points if p.values["spec"]["label"] == "off" and p.values["thinking"] is False]
    assert len(baseline) == 1
    off = commands[baseline[0].id]
    assert off.server is not None and "--speculative-algorithm" not in off.server.command
    assert main(["sweep", "--config", str(ROOT / "examples" / "sweep-qwen38-sglang.toml"), "--dry-run"]) == 0
