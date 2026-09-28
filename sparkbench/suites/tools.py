"""Tool calling: 20 exact OpenAI tool-call name/argument checks."""

from __future__ import annotations

import hashlib
import json

from sparkbench.suites.common import (
    JSONDict,
    SuiteContext,
    SuiteSpec,
    base_case,
    compact_response,
    sha256_text,
    verdict_of,
)
from sparkbench.suites.common import aggregate as aggregate_cases

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its identifier",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search internal documentation",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query", "limit"]},
        },
    },
]  # fmt: skip


def tool_cases() -> list[JSONDict]:
    rows = []
    for city, unit in [("Paris", "celsius"), ("Boston", "fahrenheit"), ("Tokyo", "celsius"), ("Oslo", "celsius"), ("Austin", "fahrenheit")]:
        rows.append({"prompt": f"What is the current weather in {city}? Use {unit}.", "name": "get_weather", "args": {"city": city, "unit": unit}})
    for expression in ["17*29", "(144/12)+8", "2**10", "91-37", "(7+5)*3"]:
        rows.append({"prompt": f"Calculate {expression} using the available tool.", "name": "calculator", "args": {"expression": expression}})
    for order in ["ORD-1042", "A-7781", "ZX-900", "2026-004", "MTP-42"]:
        rows.append({"prompt": f"Check the status of order {order}.", "name": "lookup_order", "args": {"order_id": order}})
    for query, limit in [("MTP configuration", 3), ("JSON mode", 5), ("KV cache", 2), ("tool calling", 4), ("long context", 1)]:
        rows.append({"prompt": f"Search internal docs for '{query}' and return up to {limit} results.", "name": "search_docs", "args": {"query": query, "limit": limit}})
    return rows  # fmt: skip


def normalized_tool_call(message: JSONDict) -> tuple[str | None, JSONDict | None]:
    calls = message.get("tool_calls") or []
    if not calls:
        return None, None
    function = calls[0].get("function", {})
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    return function.get("name"), arguments if isinstance(arguments, dict) else None


def run(suite: JSONDict, ctx: SuiteContext) -> None:
    rows = tool_cases()
    if ctx.smoke:
        rows = rows[:2]
    suite.setdefault("cases", [])
    done = {item["id"] for item in suite["cases"]}
    for index, row in enumerate(rows, 1):
        case_id = hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
        if case_id in done:
            continue
        response, wall = ctx.backend.chat(
            ctx.model,
            [{"role": "user", "content": row["prompt"]}],
            256,
            tools=TOOLS,
            tool_choice="auto",
        )
        compact = compact_response(response, wall)
        name, arguments = normalized_tool_call(compact["message"])
        passed = name == row["name"] and arguments == row["args"]
        suite["cases"].append({"id": case_id, "expected": row, "name": name, "arguments": arguments, "passed": passed, "response": compact})
        ctx.checkpoint()
        ctx.log(f"Tools {index}/{len(rows)} passed={passed}")
    finalize(suite)
    ctx.checkpoint()


def finalize(suite: JSONDict) -> None:
    cases = suite["cases"]
    suite["score"] = sum(item["passed"] for item in cases) / len(cases)
    suite["aggregate"] = aggregate_cases(cases)


def convert_case(case: JSONDict, suite: JSONDict) -> JSONDict:
    expected = case.get("expected") or {}
    record = base_case(
        case["id"],
        case["response"],
        verdict=verdict_of(case.get("passed")),
        prompt_sha256=sha256_text(expected["prompt"]) if "prompt" in expected else None,
    )
    record["expected"] = {"name": expected.get("name"), "arguments": expected.get("args")}
    record["predicted"] = {"name": case.get("name"), "arguments": case.get("arguments")}
    return record


def summarize(suite: JSONDict) -> tuple[float | None, str, JSONDict]:
    return suite.get("score"), "exact", {"score": suite.get("score")}


SPEC = SuiteSpec("tools", run, convert_case, summarize, "20 exact OpenAI tool-call name/argument checks")
