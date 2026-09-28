"""IFEval: 50 fixed random prompts, official Google strict/loose scorer.

The scorer is Google's ``instruction_following_eval`` package (not vendored;
``fetch_data.py``/README describe how to clone it). Its directory is taken
from ``[data] ifeval_lib``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from sparkbench.suites.common import (
    JSONDict,
    SuiteContext,
    SuiteSpec,
    base_case,
    compact_response,
    read_json,
    sha256_text,
    verdict_of,
)
from sparkbench.suites.common import aggregate as aggregate_cases


def ifeval_max_tokens(row: JSONDict) -> int:
    minimum_words = 0
    for instruction, kwargs in zip(row["instruction_id_list"], row["kwargs"], strict=False):
        if instruction == "length_constraints:number_words" and kwargs.get("relation") == "at least":
            minimum_words = max(minimum_words, int(kwargs.get("num_words", 0)))
    return min(3072, max(768, minimum_words * 3 + 256))


def load_evaluation_lib(lib_root: Path) -> ModuleType:
    if str(lib_root) not in sys.path:
        sys.path.insert(0, str(lib_root))
    return importlib.import_module("instruction_following_eval.evaluation_lib")


def score_ifeval(cases: list[JSONDict], source_rows: list[JSONDict], evaluation_lib: Any) -> JSONDict:
    by_prompt = {item["prompt"]: item["response"]["content"] for item in cases}
    inputs = [
        evaluation_lib.InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        for row in source_rows
    ]
    result: dict[str, Any] = {}
    for name, scorer in (
        ("strict", evaluation_lib.test_instruction_following_strict),
        ("loose", evaluation_lib.test_instruction_following_loose),
    ):
        outputs = [scorer(item, by_prompt) for item in inputs]
        prompt_correct = sum(item.follow_all_instructions for item in outputs)
        instruction_total = sum(len(item.follow_instruction_list) for item in outputs)
        instruction_correct = sum(sum(item.follow_instruction_list) for item in outputs)
        result[name] = {
            "prompt_level": prompt_correct / len(outputs),
            "instruction_level": instruction_correct / instruction_total,
            "prompt_correct": prompt_correct,
            "prompt_total": len(outputs),
            "instruction_correct": instruction_correct,
            "instruction_total": instruction_total,
            "details": [
                {
                    "key": inp.key,
                    "follow_all": out.follow_all_instructions,
                    "follow_list": out.follow_instruction_list,
                }
                for inp, out in zip(inputs, outputs, strict=True)
            ],
        }
    return result


def run(suite: JSONDict, ctx: SuiteContext) -> None:
    rows = read_json(ctx.data_dir / "ifeval_sample_50.json")
    if ctx.smoke:
        rows = rows[:2]
    suite.setdefault("cases", [])
    done = {item["id"] for item in suite["cases"]}
    for index, row in enumerate(rows, 1):
        case_id = str(row["key"])
        if case_id in done:
            continue
        response, wall = ctx.backend.chat(
            ctx.model,
            [{"role": "user", "content": row["prompt"]}],
            ifeval_max_tokens(row),
        )
        suite["cases"].append(
            {
                "id": case_id,
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "response": compact_response(response, wall),
            }
        )
        ctx.checkpoint()
        ctx.log(f"IFEval {index}/{len(rows)}")
    lib = ctx.options.get("evaluation_lib") or load_evaluation_lib(ctx.ifeval_lib)
    suite["scores"] = score_ifeval(suite["cases"], rows, lib)
    suite["aggregate"] = aggregate_cases(suite["cases"])
    ctx.checkpoint()


def _details(suite: JSONDict, mode: str) -> dict[str, JSONDict]:
    details = suite.get("scores", {}).get(mode, {}).get("details", [])
    return {str(item["key"]): item for item in details}


def convert_case(case: JSONDict, suite: JSONDict) -> JSONDict:
    strict = _details(suite, "strict").get(str(case["id"]))
    loose = _details(suite, "loose").get(str(case["id"]))
    record = base_case(
        case["id"],
        case["response"],
        # Verdict = official strict prompt-level result for this prompt.
        verdict=verdict_of(strict["follow_all"] if strict else None),
        # The prompt is sent verbatim as the only user message.
        prompt_sha256=case.get("prompt_sha256") or sha256_text(case["prompt"]),
    )
    record["detail"] = {
        "instruction_id_list": case.get("instruction_id_list"),
        "strict_follow_list": strict["follow_list"] if strict else None,
        "loose_follow_all": loose["follow_all"] if loose else None,
        "loose_follow_list": loose["follow_list"] if loose else None,
    }
    return record


def summarize(suite: JSONDict) -> tuple[float | None, str, JSONDict]:
    scores = suite.get("scores", {})
    metrics: JSONDict = {
        mode: {key: value for key, value in scores[mode].items() if key != "details"}
        for mode in ("strict", "loose")
        if mode in scores
    }
    quality = scores["strict"]["prompt_level"] if "strict" in scores else None
    return quality, "strict prompt", metrics


SPEC = SuiteSpec("ifeval", run, convert_case, summarize, "50 fixed random prompts, official Google strict/loose scorer")
