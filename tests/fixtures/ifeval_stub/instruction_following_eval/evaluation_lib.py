"""Minimal stand-in for Google's IFEval ``evaluation_lib`` (tests only).

Implements the same interface (InputExample, OutputExample and the strict
and loose scorer functions) for three instruction types.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, Any]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def _check(instruction: str, kwargs: dict[str, Any], response: str) -> bool:
    if instruction == "change_case:english_lowercase":
        return response == response.lower()
    if instruction == "detectable_content:postscript":
        return kwargs.get("postscript_marker", "P.S.") in response
    if instruction == "length_constraints:number_words":
        words = len(response.split())
        if kwargs.get("relation") == "at least":
            return words >= int(kwargs["num_words"])
        return words < int(kwargs["num_words"])
    raise ValueError(instruction)


def _score(inp: InputExample, response: str) -> OutputExample:
    follow = [bool(response.strip()) and _check(i, k, response) for i, k in zip(inp.instruction_id_list, inp.kwargs, strict=True)]
    return OutputExample(inp.instruction_id_list, inp.prompt, response, all(follow), follow)


def test_instruction_following_strict(inp: InputExample, prompt_to_response: dict[str, str]) -> OutputExample:
    return _score(inp, prompt_to_response[inp.prompt])


def test_instruction_following_loose(inp: InputExample, prompt_to_response: dict[str, str]) -> OutputExample:
    response = prompt_to_response[inp.prompt]
    variants = [response, response.replace("*", ""), "\n".join(response.split("\n")[1:])]
    results = [_score(inp, variant) for variant in variants]
    follow = [any(result.follow_instruction_list[i] for result in results) for i in range(len(inp.instruction_id_list))]
    return OutputExample(inp.instruction_id_list, inp.prompt, response, all(follow), follow)
