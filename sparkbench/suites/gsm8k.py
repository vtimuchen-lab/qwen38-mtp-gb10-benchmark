"""GSM8K: 100 fixed random test problems, 8-shot CoT."""

from __future__ import annotations

import hashlib
import re

from sparkbench.suites.common import (
    JSONDict,
    SuiteContext,
    SuiteSpec,
    base_case,
    compact_response,
    read_json,
    verdict_of,
)
from sparkbench.suites.common import aggregate as aggregate_cases


def normalize_number(value: str) -> str | None:
    cleaned = value.replace(",", "").replace("$", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    number = match.group(0)
    try:
        numeric = float(number)
    except ValueError:
        return None
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def gsm_answer(text: str) -> str | None:
    hashes = re.findall(r"####\s*([^\n]+)", text)
    if hashes:
        return normalize_number(hashes[-1])
    finals = re.findall(
        r"(?:final answer|answer)\s*(?:is|:|：)?\s*([^\n.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if finals:
        return normalize_number(finals[-1])
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else None


def build_prompt(question: str, shots: list[JSONDict]) -> str:
    demonstrations = "\n\n".join(
        f"Question: {item['question']}\nSolution: {item['answer']}" for item in shots
    )
    return (
        "Solve the final grade-school math problem step by step. End with the "
        "exact marker `#### number`.\n\n"
        + demonstrations
        + f"\n\nFINAL PROBLEM:\nQuestion: {question}\nSolution:"
    )


def run(suite: JSONDict, ctx: SuiteContext) -> None:
    test = read_json(ctx.data_dir / "gsm8k_test_sample_100.json")
    shots = read_json(ctx.data_dir / "gsm8k_train_shots_8.json")
    if ctx.smoke:
        test = test[:2]
    suite.setdefault("cases", [])
    done = {item["id"] for item in suite["cases"]}
    for index, row in enumerate(test, 1):
        case_id = hashlib.sha256(row["question"].encode()).hexdigest()[:16]
        if case_id in done:
            continue
        prompt = build_prompt(row["question"], shots)
        response, wall = ctx.backend.chat(ctx.model, [{"role": "user", "content": prompt}], 512)
        result = compact_response(response, wall)
        predicted = gsm_answer(result["content"])
        expected = gsm_answer(row["answer"])
        suite["cases"].append(
            {
                "id": case_id,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": result,
            }
        )
        ctx.checkpoint()
        ctx.log(f"GSM8K {index}/{len(test)} correct={predicted == expected}")
    finalize(suite)
    ctx.checkpoint()


def finalize(suite: JSONDict) -> None:
    cases = suite["cases"]
    suite["score"] = sum(item["correct"] for item in cases) / len(cases)
    suite["aggregate"] = aggregate_cases(cases)


def convert_case(case: JSONDict, suite: JSONDict) -> JSONDict:
    record = base_case(
        case["id"],
        case["response"],
        verdict=verdict_of(case.get("correct")),
        prompt_sha256=case.get("prompt_sha256"),
    )
    record["expected"] = case.get("expected")
    record["predicted"] = case.get("predicted")
    return record


def summarize(suite: JSONDict) -> tuple[float | None, str, JSONDict]:
    return suite.get("score"), "accuracy", {"score": suite.get("score")}


SPEC = SuiteSpec("gsm8k", run, convert_case, summarize, "100 fixed random test problems, 8-shot CoT")
