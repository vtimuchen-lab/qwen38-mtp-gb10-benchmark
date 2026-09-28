"""HumanEval: 40 fixed random problems, official tests in a networkless container."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

from sparkbench.suites.common import (
    Executor,
    JSONDict,
    SuiteContext,
    SuiteSpec,
    base_case,
    compact_response,
    verdict_of,
)
from sparkbench.suites.common import aggregate as aggregate_cases

DEFAULT_IMAGE = "python:3.12-slim"
PASS_MARKER = "__HUMANEVAL_PASS__"


def clean_code(text: str) -> str:
    stripped = text.strip("\n")
    fenced = re.findall(r"```(?:python)?\s*\n?(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = str(max(fenced, key=len)).strip("\n")
    stripped = re.sub(r"^(?:Here(?:'s| is).*?:)\s*", "", stripped, flags=re.IGNORECASE)
    return stripped


def build_program(problem: JSONDict, completion: str) -> str:
    code = clean_code(completion)
    entry = problem["entry_point"]
    prompt: str = problem["prompt"]
    if re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry)}\s*\(", code, re.MULTILINE):
        prefix = prompt.split("def ", 1)[0]
        program = prefix + code
    else:
        # Chat APIs commonly trim indentation only from the first generated line.
        # Restore that function-body indent while preserving nested lines verbatim.
        program = prompt + "    " + code.lstrip()
    test: str = problem["test"]
    program += "\n" + test + f"\ncheck({entry})\nprint('{PASS_MARKER}')\n"
    return program


def docker_command(image: str = DEFAULT_IMAGE) -> list[str]:
    return [
        "docker", "run", "--rm", "-i", "--pull", "never",
        "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
        "--user", "65534:65534", image, "python", "-I", "-",
    ]  # fmt: skip


def docker_executor(image: str = DEFAULT_IMAGE) -> Executor:
    def execute(program: str, timeout: float) -> tuple[bool, str]:
        try:
            completed = subprocess.run(
                docker_command(image),
                input=program,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = completed.stdout[-4000:]
            return completed.returncode == 0 and PASS_MARKER in output, output
        except subprocess.TimeoutExpired:
            return False, "timeout"

    return execute


def check_completion(problem: JSONDict, completion: str, executor: Executor, timeout: float = 8) -> tuple[bool, str]:
    return executor(build_program(problem, completion), timeout)


def load_humaneval(path: Path) -> list[JSONDict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(problem: JSONDict) -> str:
    return (
        "Complete the Python function below. Return only the code that follows "
        "the supplied prompt (normally the indented function body). Do not use "
        "Markdown fences and do not explain.\n\n" + str(problem["prompt"])
    )


def select_problems(problems: list[JSONDict], seed: int) -> list[JSONDict]:
    return random.Random(seed).sample(problems, 40)


def run(suite: JSONDict, ctx: SuiteContext) -> None:
    problems = select_problems(load_humaneval(ctx.humaneval_path), ctx.seed)
    if ctx.smoke:
        problems = problems[:2]
    if ctx.options.get("executor", "docker") != "docker":
        raise ValueError("[suites.humaneval] executor: only 'docker' (networkless sandbox) is supported")
    executor = ctx.executor or docker_executor(str(ctx.options.get("docker_image", DEFAULT_IMAGE)))
    timeout = float(ctx.options.get("timeout", 8))
    suite.setdefault("cases", [])
    done = {item["id"] for item in suite["cases"]}
    for index, problem in enumerate(problems, 1):
        case_id = problem["task_id"]
        if case_id in done:
            continue
        prompt = build_prompt(problem)
        response, wall = ctx.backend.chat(ctx.model, [{"role": "user", "content": prompt}], 512)
        result = compact_response(response, wall)
        passed, execution = check_completion(problem, result["content"], executor, timeout)
        suite["cases"].append(
            {
                "id": case_id,
                "passed": passed,
                "execution": execution,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": result,
            }
        )
        ctx.checkpoint()
        ctx.log(f"HumanEval {index}/{len(problems)} passed={passed}")
    finalize(suite)
    ctx.checkpoint()


def finalize(suite: JSONDict) -> None:
    cases = suite["cases"]
    suite["pass_at_1"] = sum(item["passed"] for item in cases) / len(cases)
    suite["aggregate"] = aggregate_cases(cases)


def convert_case(case: JSONDict, suite: JSONDict) -> JSONDict:
    record = base_case(
        case["id"],
        case["response"],
        verdict=verdict_of(case.get("passed")),
        prompt_sha256=case.get("prompt_sha256"),
    )
    record["detail"] = {"execution": case.get("execution")}
    return record


def summarize(suite: JSONDict) -> tuple[float | None, str, JSONDict]:
    return suite.get("pass_at_1"), "pass@1", {"pass_at_1": suite.get("pass_at_1")}


SPEC = SuiteSpec(
    "humaneval",
    run,
    convert_case,
    summarize,
    "40 fixed random problems, chat completion, official tests in networkless read-only Docker",
)
