"""Reports over the imported results/v1 files reproduce the published numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import ROOT

from sparkbench.jsonutil import read_json, sha256_file
from sparkbench.report import report

V1 = ROOT / "results" / "v1"


@pytest.fixture(scope="module")
def q4() -> dict:
    return read_json(V1 / "main_q4_mtp7.json")


@pytest.fixture(scope="module")
def q6() -> dict:
    return read_json(V1 / "main_q6_mtp7.json")


def test_readme_headline_numbers(q4: dict, q6: dict) -> None:
    summary, markdown = report(q4, q6)
    assert summary["a"]["overall"]["decode_tokens_per_second"] == pytest.approx(21.48, abs=0.01)
    assert summary["b"]["overall"]["decode_tokens_per_second"] == pytest.approx(17.43, abs=0.01)
    assert round(summary["speed"]["overall"]["a_vs_b"] * 100, 1) == 23.2
    assert "q4-mtp7 is 23.2% faster than q6-mtp7 (q6-mtp7 as denominator)." in markdown
    for doc in (q4, q6):
        assert doc["suites"]["long_context"]["verdicts"] == {"pass": 9, "fail": 0, "error": 0, "unscored": 0}
        assert doc["suites"]["tools"]["verdicts"]["pass"] == 20 and doc["suites"]["tools"]["aggregate"]["cases"] == 20
        assert doc["suites"]["stability_100"]["verdicts"]["pass"] == 100
        assert doc["suites"]["stability_100"]["metrics"]["unique_outputs"] == 1
    assert "- Long-context exact: q4-mtp7 9/9, q6-mtp7 9/9." in markdown
    assert "- Tool calls: q4-mtp7 20/20, q6-mtp7 20/20." in markdown
    assert "q4-mtp7 100/100 exact (1 unique output(s)), q6-mtp7 100/100 exact" in markdown


def test_matches_make_report_summary(q4: dict, q6: dict) -> None:
    """Every number of the legacy results/summary.json is reproduced."""
    legacy = read_json(ROOT / "results" / "summary.json")
    summary, _ = report(q4, q6)
    for side, key in (("a", "q4"), ("b", "q6")):
        for field, value in legacy[key]["overall"].items():
            assert summary[side]["overall"][field] == pytest.approx(value, rel=1e-12), field
        for name, metrics in legacy[key]["suites"].items():
            row = summary[side]["suites"][name]
            assert row["quality"] == pytest.approx(metrics["quality"])
            assert row["decode_tokens_per_second"] == metrics["decode_tokens_per_second"]
            assert row["prompt_tokens_per_second"] == metrics["prompt_tokens_per_second"]
            assert row["draft_acceptance"] == metrics["draft_acceptance"]
    for name, legacy_name in (("mmlu_pro", "mmlu_pro"), ("gsm8k", "gsm8k"), ("humaneval", "humaneval"), ("ifeval", "ifeval_strict_prompt")):
        old = legacy["paired"][legacy_name]
        new = summary["paired"][name]
        assert (new["both_correct"], new["a_only"], new["b_only"], new["both_wrong"]) == (old["both_correct"], old["q4_only"], old["q6_only"], old["both_wrong"])
        assert new["mcnemar_exact_p"] == pytest.approx(old["mcnemar_exact_p"])
    for category, scores in legacy["mmlu_categories"].items():
        assert summary["mmlu_categories"][category] == {"a": scores["q4"], "b": scores["q6"]}
    ifeval_q4 = q4["suites"]["ifeval"]["metrics"]
    assert ifeval_q4["strict"]["instruction_level"] == legacy["q4"]["suites"]["ifeval"]["strict_instruction"]
    assert ifeval_q4["loose"]["prompt_level"] == legacy["q4"]["suites"]["ifeval"]["loose_prompt"]


def test_suite_aggregates_bit_identical_to_legacy(q4: dict, q6: dict) -> None:
    for doc, name in ((q4, "q4"), (q6, "q6")):
        legacy = read_json(ROOT / "results" / f"{name}.json")
        for suite, data in legacy["suites"].items():
            assert doc["suites"][suite]["aggregate"] == data["aggregate"], (name, suite)


def test_source_results_unchanged() -> None:
    """Hashes published in README 'Result integrity'."""
    assert sha256_file(ROOT / "results" / "q4.json") == "06d328340309732e66be2d77c2c15df8e8f164c96a5be07aabdb0e883cdfd12a"
    assert sha256_file(ROOT / "results" / "q6.json") == "b20917c30186f77a513e69f368a1ba2dfa842d4d70de94724178f68536ab472e"
    assert sha256_file(ROOT / "results" / "summary.json") == "76081f2001c378f954317f41f1bdc39184d0bb4020ab007cf04e3f0574436523"


def test_sweep_and_series_aggregates_reproduce_legacy() -> None:
    sweep = read_json(ROOT / "mtp_sweep" / "results.json")
    for mode, state in sweep["modes"].items():
        doc = read_json(V1 / f"mtp_sweep_q4_{mode}.json")
        aggregate = doc["suites"]["predictable_9"]["aggregate"]
        for key in ("prompt_tokens", "generated_tokens", "draft_generated", "draft_accepted"):
            assert aggregate[key] == state["aggregate"][key], (mode, key)
        for key in ("decode_tokens_per_second", "prompt_tokens_per_second", "median_case_decode_tokens_per_second", "draft_acceptance", "wall_seconds"):
            assert aggregate[key] == pytest.approx(state["aggregate"][key], rel=1e-12), (mode, key)
    mtp10 = read_json(V1 / "mtp_sweep_q4_mtp10.json")["suites"]["predictable_9"]
    assert mtp10["aggregate"]["decode_tokens_per_second"] == pytest.approx(37.01, abs=0.01)  # README: 37.01 tok/s
    assert mtp10["verdicts"]["pass"] == 9
    complex_state = read_json(ROOT / "extended_validation" / "complex_results.json")
    for mode, state in complex_state["modes"].items():
        doc = read_json(V1 / f"complex_q4_{mode}.json")
        for name, legacy_suite in state["suites"].items():
            new = doc["suites"][f"mini_{name}"]["aggregate"]
            for key, value in legacy_suite["aggregate"].items():
                assert new[key] == pytest.approx(value, rel=1e-12), (mode, name, key)
    prefix = read_json(ROOT / "extended_validation" / "prefix_cache_results.json")
    for mode, state in prefix["modes"].items():
        suite = read_json(V1 / f"prefix_cache_q4_mtp7_{mode}.json")["suites"]["agent_session_12"]
        assert suite["aggregate"]["prompt_tokens"] == state["prompt_tokens_evaluated"]
        assert suite["aggregate"]["decode_tokens_per_second"] == pytest.approx(state["decode_tokens_per_second"], rel=1e-12)
        assert suite["verdicts"]["pass"] == state["quality_passed"]
    soak = read_json(ROOT / "soak_partial" / "summary.json")
    for quant, stats in soak["by_quant"].items():
        doc = read_json(V1 / f"soak_{quant}_mtp7.json")
        assert doc["overall"]["cases"] == stats["requests"]
        assert doc["overall"]["generated_tokens"] == stats["generated_tokens"]
        assert doc["overall"]["decode_tokens_per_second"] == pytest.approx(stats["decode_tokens_per_second"], rel=1e-9)
        assert doc["overall"]["draft_acceptance"] == pytest.approx(stats["draft_acceptance"], rel=1e-9)
        assert sum(s["verdicts"]["pass"] for s in doc["suites"].values()) == stats["passed"]


def test_committed_v1_files_are_reproducible(tmp_path: Path) -> None:
    """results/v1 is exactly what `python -m sparkbench import` produces."""
    from sparkbench.importers import import_all

    written = import_all(ROOT, tmp_path)
    committed = sorted(path.name for path in V1.glob("*.json"))
    assert sorted(path.name for path in written.values()) == committed
    assert len(committed) == 28
    for path in written.values():
        assert json.loads(path.read_text(encoding="utf-8")) == read_json(V1 / path.name), path.name
