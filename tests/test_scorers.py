"""Unit tests for every scorer on tiny fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import FIXTURES, IFEVAL_STUB, ROOT, humaneval_problems, run_python

from sparkbench.suites import gsm8k, humaneval, ifeval, long_context, mmlu_pro, stability, tools
from sparkbench.suites.common import aggregate, compact_response, merge_responses


def load(name: str) -> list[dict]:
    return json.loads((FIXTURES / "data" / name).read_text(encoding="utf-8"))


def legacy(name: str) -> dict:
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


# -- MMLU-Pro ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: D", "D"),
        ("The answer is (c).", "C"),
        ("First answer: A ... final answer: J", "J"),
        ("答案是 B", "B"),
        ("(E).", "E"),
        ("  g  ", "G"),
        ("No letter here", None),
        ("K", None),
    ],
)
def test_parse_mcq(text: str, expected: str | None) -> None:
    assert mmlu_pro.parse_mcq(text) == expected


def test_format_mcq_and_prompt() -> None:
    rows = load("mmlu_pro_validation.json")
    with_cot = mmlu_pro.format_mcq(rows[0], True)
    assert with_cot.endswith("The answer is (B).\nAnswer: B")  # CoT lacks "Answer: X" -> appended
    assert mmlu_pro.format_mcq(rows[2], True).endswith("B. 2\nAnswer: A")
    assert mmlu_pro.format_mcq(rows[1], False) == "2 + 2 = ?\nA. 3\nB. 4\nC. 5"
    shots = {"math": rows[1:]}
    prompt = mmlu_pro.build_prompt(load("mmlu_pro_test_stratified_100.json")[1], shots)
    assert prompt.startswith("Solve the final multiple-choice problem.")
    assert prompt.endswith("FINAL PROBLEM:\nWhat is 7 * 8?\nA. 54\nB. 56\nC. 58\nD. 64\nE. 48")


def test_mmlu_finalize_scores_by_category() -> None:
    def case(category: str, correct: bool) -> dict:
        return {"category": category, "correct": correct, "response": {"timings": {}, "wall_seconds": 1.0}}

    suite = {"cases": [case("math", True), case("math", False), case("law", True)]}
    mmlu_pro.finalize(suite)
    assert suite["score"] == pytest.approx(2 / 3)
    assert suite["by_category"] == {"law": 1.0, "math": 0.5}


# -- GSM8K -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\n#### 1,234", "1234"),
        ("#### $18.00", "18"),
        ("#### 2.50 dollars", "2.5"),
        ("The final answer is 42.", "42"),
        ("so we get 3 then 17", "17"),
        ("#### -5", "-5"),
        ("nothing numeric", None),
    ],
)
def test_gsm_answer(text: str, expected: str | None) -> None:
    assert gsm8k.gsm_answer(text) == expected


def test_gsm_prompt_and_reference_answers() -> None:
    shots = load("gsm8k_train_shots_8.json")
    rows = load("gsm8k_test_sample_100.json")
    prompt = gsm8k.build_prompt(rows[0]["question"], shots)
    assert "Question: Ann had 5 pens and lost 2. How many are left?\nSolution: 5 - 2" in prompt
    assert prompt.endswith(f"FINAL PROBLEM:\nQuestion: {rows[0]['question']}\nSolution:")
    assert [gsm8k.gsm_answer(row["answer"]) for row in rows] == ["7", "27", "2000"]


# -- HumanEval ---------------------------------------------------------------


def test_clean_code_variants() -> None:
    assert humaneval.clean_code("```python\nreturn 1\n```") == "return 1"
    assert humaneval.clean_code("Here is the code:\nreturn 2") == "return 2"
    assert humaneval.clean_code("\n\n    return x\n") == "    return x"


def test_build_program_body_and_full_function() -> None:
    problem = humaneval_problems(1)[0]
    body = humaneval.build_program(problem, "return x")
    assert body.startswith(problem["prompt"] + "    return x\n")
    assert body.endswith("check(f0)\nprint('__HUMANEVAL_PASS__')\n")
    full = humaneval.build_program(problem, "def f0(x):\n    return x")
    assert full.startswith("def f0(x):\n    return x\n")


def test_humaneval_execution_pass_and_fail() -> None:
    problems = humaneval_problems(8)
    assert humaneval.check_completion(problems[0], "return x", run_python) == (True, "__HUMANEVAL_PASS__\n")
    passed, output = humaneval.check_completion(problems[7], "return x", run_python)
    assert not passed and "AssertionError" in output


def test_humaneval_sample_matches_published_run() -> None:
    """random.Random(42).sample(164 problems, 40) picks the published task ids."""
    selected = humaneval.select_problems(humaneval_problems(), 42)
    published = [case["id"] for case in legacy("q4")["suites"]["humaneval"]["cases"]]
    assert [problem["task_id"] for problem in selected] == published


def test_docker_command_is_sandboxed() -> None:
    command = humaneval.docker_command()
    assert command[:3] == ["docker", "run", "--rm"]
    for flag in ("--network", "--read-only", "--cap-drop", "--pids-limit", "--memory", "--user"):
        assert flag in command
    assert command[-4:] == ["python:3.12-slim", "python", "-I", "-"]


# -- IFEval ------------------------------------------------------------------


def test_ifeval_max_tokens() -> None:
    rows = load("ifeval_sample_50.json")
    assert ifeval.ifeval_max_tokens(rows[0]) == 768
    assert ifeval.ifeval_max_tokens({"instruction_id_list": ["length_constraints:number_words"], "kwargs": [{"relation": "at least", "num_words": 300}]}) == 1156
    assert ifeval.ifeval_max_tokens({"instruction_id_list": ["length_constraints:number_words"], "kwargs": [{"relation": "at least", "num_words": 5000}]}) == 3072
    assert ifeval.ifeval_max_tokens({"instruction_id_list": ["length_constraints:number_words"], "kwargs": [{"relation": "less than", "num_words": 5000}]}) == 768


def test_score_ifeval_with_stub_lib() -> None:
    rows = load("ifeval_sample_50.json")
    lib = ifeval.load_evaluation_lib(IFEVAL_STUB)
    responses = ["all lowercase here.", "Hello\nP.S. see you", "Too short."]
    cases = [{"prompt": row["prompt"], "response": {"content": text}} for row, text in zip(rows, responses, strict=True)]
    scores = ifeval.score_ifeval(cases, rows, lib)
    assert scores["strict"]["prompt_correct"] == 2
    assert scores["strict"]["prompt_level"] == pytest.approx(2 / 3)
    assert [item["follow_all"] for item in scores["strict"]["details"]] == [True, True, False]
    assert scores["loose"]["instruction_total"] == 3


# -- Tools -------------------------------------------------------------------


def test_tool_cases_ids_match_published_run() -> None:
    ids = [hashlib.sha256(row["prompt"].encode()).hexdigest()[:16] for row in tools.tool_cases()]
    published = [case["id"] for case in legacy("q4")["suites"]["tools"]["cases"]]
    assert ids == published
    assert len(tools.TOOLS) == 4


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"tool_calls": [{"function": {"name": "calculator", "arguments": '{"expression": "2**10"}'}}]}, ("calculator", {"expression": "2**10"})),
        ({"tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "1"}}}]}, ("calculator", {"expression": "1"})),
        ({"tool_calls": [{"function": {"name": "calculator", "arguments": "{broken"}}]}, ("calculator", None)),
        ({"tool_calls": [{"function": {"name": "x", "arguments": "[1, 2]"}}]}, ("x", None)),
        ({"content": "no tools"}, (None, None)),
    ],
)
def test_normalized_tool_call(message: dict, expected: tuple) -> None:
    assert tools.normalized_tool_call(message) == expected


# -- Long context ------------------------------------------------------------


def test_score_long() -> None:
    secret = "CTX8000_start_K9Q7"
    assert long_context.score_long("needle", f"  {secret}\n", secret)
    assert not long_context.score_long("needle", f"The secret is {secret}", secret)
    assert long_context.score_long("json", json.dumps({"owner": "Marta", "secret": secret}), secret)
    assert not long_context.score_long("json", "```json\n{}\n```", secret)
    assert long_context.score_long("code", f"def get_secret():\n    return '{secret}'", secret)
    assert not long_context.score_long("code", f"```python\ndef get_secret():\n    return '{secret}'\n```", secret)
    assert not long_context.score_long("code", "def get_secret(:", secret)


def test_build_long_prompt_hits_target() -> None:
    count = lambda text: max(1, len(text) // 4)  # noqa: E731
    prompt, tokens, secret = long_context.build_long_prompt(8000, "middle", "json", count)
    assert secret == "CTX8000_middle_K9Q7"
    assert 7800 <= tokens <= 8000
    assert prompt.index("ACTIVE_NEEDLE") == pytest.approx(len(prompt) / 2, rel=0.05)
    assert [f"{t}_{p}_{k}" for t, p, k in long_context.definitions()] == [
        case["id"] for case in legacy("q4")["suites"]["long_context"]["cases"]
    ]


# -- Stability and shared helpers --------------------------------------------


def test_stability_finalize() -> None:
    def case(text: str) -> dict:
        return {"content_sha256": hashlib.sha256(text.encode()).hexdigest(), "exact": text == stability.EXPECTED, "response": {}}

    suite = {"cases": [case("BENCHMARK_OK_42")] * 3 + [case("other")]}
    stability.finalize(suite)
    assert suite["exact_rate"] == 0.75
    assert suite["unique_outputs"] == 2
    assert suite["modal_output_rate"] == 0.75


def test_merge_and_aggregate() -> None:
    first = {"timings": {"prompt_n": 10, "prompt_ms": 100.0, "predicted_n": 20, "predicted_ms": 1000.0, "draft_n": 30, "draft_n_accepted": 15}, "wall_seconds": 1.5}
    second = {"content": "Answer: A", "timings": {"prompt_n": 40, "prompt_ms": 50.0, "predicted_n": 4, "predicted_ms": 200.0}, "wall_seconds": 0.5}
    merged = merge_responses(first, second)
    assert merged["timings"]["predicted_n"] == 24
    assert merged["timings"]["predicted_per_second"] == pytest.approx(20.0)
    assert merged["wall_seconds"] == 2.0
    result = aggregate([{"response": merged}])
    assert result["decode_tokens_per_second"] == pytest.approx(20.0)
    assert result["prompt_tokens_per_second"] == pytest.approx(50 / 0.15)
    assert result["draft_acceptance"] == 0.5


def test_compact_response_without_server_timings() -> None:
    response = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 10}}
    compact = compact_response(response, 2.0)
    assert compact["timings"] == {"source": "wall", "prompt_n": 5, "predicted_n": 10, "predicted_ms": 2000.0, "predicted_per_second": 5.0}


def test_fixtures_are_small() -> None:
    total = sum(path.stat().st_size for path in FIXTURES.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    assert total < 16_000
    assert not any(Path(FIXTURES).rglob("*.gz"))
