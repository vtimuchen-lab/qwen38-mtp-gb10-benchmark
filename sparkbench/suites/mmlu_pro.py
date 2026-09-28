"""MMLU-Pro: 100 category-stratified questions, 5-shot CoT, two-stage answer."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from sparkbench.suites.common import (
    JSONDict,
    SuiteContext,
    SuiteSpec,
    base_case,
    compact_response,
    merge_responses,
    read_json,
    verdict_of,
)
from sparkbench.suites.common import aggregate as aggregate_cases

LETTERS = "ABCDEFGHIJ"


def format_mcq(row: JSONDict, include_answer: bool) -> str:
    lines = [row["question"]]
    lines.extend(f"{LETTERS[i]}. {option}" for i, option in enumerate(row["options"]))
    if include_answer:
        cot = (row.get("cot_content") or "").strip()
        if cot:
            lines.append(cot)
        if not re.search(r"answer\s*[:：].*[A-J]", cot, re.IGNORECASE):
            lines.append(f"Answer: {row['answer']}")
    return "\n".join(lines)


def parse_mcq(text: str) -> str | None:
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:|：)?\s*\(?([A-J])\)?",
        r"答案\s*(?:是|:|：)?\s*\(?([A-J])\)?",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    if matches:
        return matches[-1].upper()
    direct = re.fullmatch(r"\s*\(?([A-J])\)?[.)]?\s*", text, flags=re.IGNORECASE)
    return direct.group(1).upper() if direct else None


FINAL_INSTRUCTION = (
    "Now give only the final choice in exactly the form `Answer: X`, where X is A-J. No explanation."
)


def build_prompt(row: JSONDict, shots: dict[str, list[JSONDict]]) -> str:
    examples = "\n\n".join(format_mcq(item, True) for item in shots[row["category"]][:5])
    return (
        "Solve the final multiple-choice problem. Think carefully and finish with "
        "a separate line in exactly the form `Answer: X`, where X is A-J.\n\n"
        + examples
        + "\n\nFINAL PROBLEM:\n"
        + format_mcq(row, False)
    )


def run(suite: JSONDict, ctx: SuiteContext) -> None:
    test = read_json(ctx.data_dir / "mmlu_pro_test_stratified_100.json")
    validation = read_json(ctx.data_dir / "mmlu_pro_validation.json")
    if ctx.smoke:
        test = test[:2]
    shots: dict[str, list[JSONDict]] = defaultdict(list)
    for row in validation:
        shots[row["category"]].append(row)
    suite.setdefault("cases", [])
    # A response cut exactly at the generation cap has no trustworthy final
    # choice. Drop such checkpoints so a resumed run recomputes them.
    suite["cases"] = [
        item for item in suite["cases"] if item.get("response", {}).get("finish_reason") != "length"
    ]
    done = {item["id"] for item in suite["cases"]}
    for index, row in enumerate(test, 1):
        case_id = str(row["question_id"])
        if case_id in done:
            continue
        prompt = build_prompt(row, shots)
        reasoning_response, reasoning_wall = ctx.backend.chat(
            ctx.model,
            [{"role": "user", "content": prompt}],
            1024,
        )
        reasoning = compact_response(reasoning_response, reasoning_wall)
        final_response, final_wall = ctx.backend.chat(
            ctx.model,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning["content"]},
                {"role": "user", "content": FINAL_INSTRUCTION},
            ],
            32,
        )
        final = compact_response(final_response, final_wall)
        result = merge_responses(reasoning, final)
        predicted = parse_mcq(final["content"])
        suite["cases"].append(
            {
                "id": case_id,
                "category": row["category"],
                "expected": row["answer"],
                "predicted": predicted,
                "correct": predicted == row["answer"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "reasoning_response": reasoning,
                "response": result,
            }
        )
        ctx.checkpoint()
        ctx.log(f"MMLU-Pro {index}/{len(test)} correct={predicted == row['answer']}")
    finalize(suite)
    ctx.checkpoint()


def finalize(suite: JSONDict) -> None:
    cases = suite["cases"]
    suite["score"] = sum(item["correct"] for item in cases) / len(cases)
    suite["by_category"] = {
        category: sum(item["correct"] for item in cases if item["category"] == category)
        / sum(1 for item in cases if item["category"] == category)
        for category in sorted({item["category"] for item in cases})
    }
    suite["aggregate"] = aggregate_cases(cases)


def convert_case(case: JSONDict, suite: JSONDict) -> JSONDict:
    reasoning = case.get("reasoning_response") or {}
    record = base_case(
        case["id"],
        case["response"],
        verdict=verdict_of(case.get("correct")),
        prompt_sha256=case.get("prompt_sha256"),
        api_calls=2,
        transcript=[str(reasoning.get("content", ""))],
    )
    record["expected"] = case.get("expected")
    record["predicted"] = case.get("predicted")
    record["detail"] = {"category": case.get("category")}
    return record


def summarize(suite: JSONDict) -> tuple[float | None, str, JSONDict]:
    metrics: dict[str, Any] = {"score": suite.get("score"), "by_category": suite.get("by_category", {})}
    return suite.get("score"), "accuracy", metrics


SPEC = SuiteSpec(
    "mmlu_pro",
    run,
    convert_case,
    summarize,
    "100 category-stratified, 5-shot CoT, 1024-token reasoning plus deterministic final-letter extraction",
)
