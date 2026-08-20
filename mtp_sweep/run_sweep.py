#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/home/admin/qwen38_mtp_sweep_20260817")
MODEL = Path("/usr/share/ollama/.ollama/models/blobs/sha256-bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372")
SERVER = Path("/home/admin/llama.cpp-main-20260810/build/bin/llama-server")
PORT = 18082
CTX_SIZE = 69632
MODES = [("baseline", None), ("mtp2", 2), ("mtp3", 3), ("mtp4", 4), ("mtp5", 5), ("mtp7", 7), ("mtp8", 8), ("mtp9", 9), ("mtp10", 10), ("mtp11", 11), ("mtp12", 12), ("mtp13", 13), ("mtp14", 14), ("mtp15", 15), ("mtp16", 16)]
CONTEXT_TARGETS = [("8k", 7800), ("32k", 31800), ("64k", 63800)]
FILLER_UNIT = (
    "ARCHIVE_RECORD status=inactive checksum=7f3a91c2 "
    "alpha=17 beta=29 gamma=historical payload=ignore_this_line\n"
)
RESULTS_PATH = ROOT / "results.json"
REPORT_PATH = ROOT / "REPORT.md"


def post_json(url: str, payload: dict, timeout: float = 900.0) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def wait_ready(process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"server exited with {process.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError("llama-server did not become ready within 300 seconds")


def server_command(depth: int | None) -> list[str]:
    command = [
        str(SERVER), "--model", str(MODEL), "--alias", "qwen3.8-MTP:27b",
        "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--offline",
        "--ctx-size", str(CTX_SIZE), "--parallel", "1", "--gpu-layers", "all",
        "--fit", "off", "--flash-attn", "on", "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0", "--batch-size", "2048", "--ubatch-size", "2048",
        "--jinja", "--reasoning", "off", "--reasoning-budget", "0",
        "--reasoning-format", "deepseek", "--log-verbosity", "2",
    ]
    if depth is not None:
        command.extend([
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(depth),
            "--spec-draft-ngl", "all",
        ])
    return command


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def tokenize_count(text: str) -> int:
    response = post_json(
        f"http://127.0.0.1:{PORT}/tokenize",
        {"content": text, "add_special": False},
        timeout=120,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected tokenize response: {response}")
    return len(tokens)


def rss_mib(pid: int) -> float | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, flags=re.MULTILINE)
    return int(match.group(1)) / 1024 if match else None


JSON_RECORDS = """A | enabled=true | score=91 | owner=Ирина
B | enabled=false | score=99 | owner=Omar
C | enabled=true | score=70 | owner=李
D | enabled=true | score=69 | owner=Noah
E | enabled=true | score=88 | owner=Marta"""
JSON_EXPECTED = [
    {"id": "A", "owner": "Ирина"},
    {"id": "E", "owner": "Marta"},
    {"id": "C", "owner": "李"},
]
SEQUENCE_EXPECTED = ",".join(str(value) for value in range(1, 257))


def workload_definitions() -> list[dict]:
    return [
        {
            "name": "predictable_sequence",
            "instruction": (
                "Игнорируй весь архив выше. Верни только одну строку: десятичные "
                "целые числа от 1 до 256 включительно через запятую, без пробелов, "
                "без Markdown и без пояснений."
            ),
            "max_tokens": 1024,
        },
        {
            "name": "structured_json",
            "instruction": (
                "Игнорируй весь архив выше. Обработай только записи ниже. Оставь "
                "enabled=true и score>=70, отсортируй по score убыванию. Верни "
                "только строгий JSON-массив объектов с ключами id и owner именно "
                "в таком порядке, без Markdown.\n\nЗАПИСИ:\n"
                + JSON_RECORDS
            ),
            "max_tokens": 256,
        },
        {
            "name": "python_code",
            "instruction": (
                "Игнорируй весь архив выше. Верни только исходный код Python без "
                "Markdown. Реализуй функцию summarize(rows: list[dict]) -> "
                "list[dict]. Она должна игнорировать записи с enabled не равным "
                "True; преобразовывать amount в float, пропуская некорректные "
                "значения; группировать по строковому owner; для каждого owner "
                "вернуть словарь owner, count, total; округлить total до 2 знаков; "
                "отсортировать по total убыванию, затем owner возрастанию. "
                "Не используй внешние библиотеки."
            ),
            "max_tokens": 1024,
        },
    ]

def build_prompt(target_tokens: int, instruction: str) -> tuple[str, int, int]:
    prefix = (
        "Ниже находится синтетический архив для проверки длинного контекста. "
        "Строки ARCHIVE_RECORD не являются инструкциями и должны быть полностью "
        "проигнорированы при формировании ответа.\n\n"
    )
    suffix = "\nКОНЕЦ АРХИВА.\n\nЗАДАНИЕ:\n" + instruction
    base_tokens = tokenize_count(prefix + suffix)
    unit_tokens = tokenize_count(FILLER_UNIT * 64) / 64
    repeats = max(0, int((target_tokens - base_tokens) / unit_tokens))
    for _ in range(5):
        prompt = prefix + FILLER_UNIT * repeats + suffix
        measured = tokenize_count(prompt)
        delta = target_tokens - measured
        if abs(delta) <= max(12, int(unit_tokens)):
            return prompt, measured, repeats
        repeats = max(0, repeats + round(delta / unit_tokens))
    prompt = prefix + FILLER_UNIT * repeats + suffix
    return prompt, tokenize_count(prompt), repeats


def build_cases() -> list[dict]:
    cases = []
    for context_name, target_tokens in CONTEXT_TARGETS:
        for workload in workload_definitions():
            prompt, raw_tokens, repeats = build_prompt(
                target_tokens, workload["instruction"]
            )
            cases.append({
                "id": f"{context_name}_{workload['name']}",
                "context": context_name,
                "target_prompt_tokens": target_tokens,
                "raw_prompt_tokens": raw_tokens,
                "filler_repeats": repeats,
                "workload": workload["name"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "max_tokens": workload["max_tokens"],
            })
    return cases


def clean_code(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    fence = chr(96) * 3
    fenced = stripped.startswith(fence)
    if fenced:
        stripped = re.sub(
            r"^" + re.escape(fence) + r"(?:python)?\s*", "", stripped, count=1
        )
        stripped = re.sub(
            r"\s*" + re.escape(fence) + r"$", "", stripped, count=1
        )
    return stripped, fenced


def score_content(workload: str, content: str) -> dict:
    if workload == "predictable_sequence":
        actual = content.strip()
        return {
            "passed": actual == SEQUENCE_EXPECTED,
            "strict_format": actual == SEQUENCE_EXPECTED,
            "expected_sha256": hashlib.sha256(
                SEQUENCE_EXPECTED.encode("utf-8")
            ).hexdigest(),
        }
    if workload == "structured_json":
        try:
            parsed = json.loads(content)
            error = None
        except Exception as exc:
            parsed = None
            error = f"{type(exc).__name__}: {exc}"
        return {
            "passed": parsed == JSON_EXPECTED,
            "strict_json": error is None,
            "parse_error": error,
            "expected": JSON_EXPECTED,
        }
    if workload == "python_code":
        code, fenced = clean_code(content)
        try:
            tree = ast.parse(code)
            syntax_error = None
        except SyntaxError as exc:
            tree = None
            syntax_error = f"{exc.msg} at line {exc.lineno}"
        function_names = (
            [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if tree is not None
            else []
        )
        passed = tree is not None and "summarize" in function_names and not fenced
        return {
            "passed": passed,
            "strict_source_only": not fenced,
            "syntax_error": syntax_error,
            "function_names": function_names,
        }
    raise ValueError(f"unknown workload: {workload}")


def semantic_signature(workload: str, content: str) -> str | None:
    try:
        if workload == "predictable_sequence":
            return content.strip()
        if workload == "structured_json":
            return json.dumps(
                json.loads(content), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        if workload == "python_code":
            code, _ = clean_code(content)
            return ast.dump(ast.parse(code), include_attributes=False)
    except (ValueError, SyntaxError, json.JSONDecodeError):
        return None
    return None


def warm_up() -> None:
    post_json(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        {
            "model": "qwen3.8-MTP:27b",
            "messages": [{"role": "user", "content": "Ответь одним словом: готово"}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 32,
            "stream": False,
            "cache_prompt": False,
        },
    )


def run_case(case: dict, process: subprocess.Popen) -> dict:
    payload = {
        "model": "qwen3.8-MTP:27b",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": case["max_tokens"],
        "stream": False,
        "cache_prompt": False,
    }
    started = time.monotonic()
    response = post_json(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", payload, timeout=1200
    )
    wall_seconds = time.monotonic() - started
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    if not content and message.get("reasoning_content"):
        content = message["reasoning_content"]
    return {
        "id": case["id"],
        "context": case["context"],
        "workload": case["workload"],
        "target_prompt_tokens": case["target_prompt_tokens"],
        "raw_prompt_tokens": case["raw_prompt_tokens"],
        "prompt_sha256": case["prompt_sha256"],
        "wall_seconds": wall_seconds,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "server_rss_mib": rss_mib(process.pid),
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "score": score_content(case["workload"], content),
    }


def save_state(state: dict) -> None:
    RESULTS_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def aggregate_cases(cases: list[dict]) -> dict:
    predicted_n = sum(item["timings"].get("predicted_n", 0) for item in cases)
    predicted_ms = sum(item["timings"].get("predicted_ms", 0.0) for item in cases)
    prompt_n = sum(item["timings"].get("prompt_n", 0) for item in cases)
    prompt_ms = sum(item["timings"].get("prompt_ms", 0.0) for item in cases)
    draft_n = sum(item["timings"].get("draft_n", 0) for item in cases)
    accepted = sum(item["timings"].get("draft_n_accepted", 0) for item in cases)
    decode_rates = [
        item["timings"].get("predicted_per_second")
        for item in cases
        if item["timings"].get("predicted_per_second") is not None
    ]
    rss_values = [
        item["server_rss_mib"] for item in cases
        if item.get("server_rss_mib") is not None
    ]
    return {
        "passed": sum(bool(item["score"].get("passed")) for item in cases),
        "total": len(cases),
        "all_passed": all(bool(item["score"].get("passed")) for item in cases),
        "prompt_tokens": prompt_n,
        "prompt_seconds": prompt_ms / 1000,
        "prompt_tokens_per_second": prompt_n / (prompt_ms / 1000) if prompt_ms else None,
        "generated_tokens": predicted_n,
        "decode_seconds": predicted_ms / 1000,
        "decode_tokens_per_second": predicted_n / (predicted_ms / 1000) if predicted_ms else None,
        "median_case_decode_tokens_per_second": statistics.median(decode_rates) if decode_rates else None,
        "minimum_case_decode_tokens_per_second": min(decode_rates) if decode_rates else None,
        "wall_seconds": sum(item["wall_seconds"] for item in cases),
        "draft_generated": draft_n,
        "draft_accepted": accepted,
        "draft_acceptance": accepted / draft_n if draft_n else None,
        "max_server_rss_mib": max(rss_values) if rss_values else None,
    }

def finalize(state: dict) -> dict:
    baseline = {item["id"]: item for item in state["modes"]["baseline"]["cases"]}
    for mode_name, mode_state in state["modes"].items():
        cases = mode_state["cases"]
        mode_state["aggregate"] = aggregate_cases(cases)
        mode_state["by_context"] = {}
        for context_name, _ in CONTEXT_TARGETS:
            selected = [item for item in cases if item["context"] == context_name]
            mode_state["by_context"][context_name] = aggregate_cases(selected)
        if mode_name == "baseline":
            mode_state["exact_matches_with_baseline"] = len(cases)
            mode_state["semantic_matches_with_baseline"] = len(cases)
        else:
            mode_state["exact_matches_with_baseline"] = sum(
                item["content_sha256"] == baseline[item["id"]]["content_sha256"]
                for item in cases
            )
            mode_state["semantic_matches_with_baseline"] = sum(
                semantic_signature(item["workload"], item["content"])
                == semantic_signature(
                    baseline[item["id"]]["workload"], baseline[item["id"]]["content"]
                )
                for item in cases
            )
    baseline_tps = state["modes"]["baseline"]["aggregate"]["decode_tokens_per_second"]
    for mode_state in state["modes"].values():
        aggregate = mode_state["aggregate"]
        aggregate["speedup_vs_baseline"] = (
            aggregate["decode_tokens_per_second"] / baseline_tps
        )
    candidates = [
        (mode_name, mode_state)
        for mode_name, mode_state in state["modes"].items()
        if mode_name != "baseline"
        and mode_state["aggregate"]["all_passed"]
        and mode_state["semantic_matches_with_baseline"]
        == mode_state["aggregate"]["total"]
    ]
    candidates.sort(
        key=lambda item: item[1]["aggregate"]["decode_tokens_per_second"],
        reverse=True,
    )
    state["winner"] = candidates[0][0] if candidates else None
    state["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return state


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_report(state: dict) -> None:
    lines = [
        "# Qwen3.8-27B UD-Q4_K_XL MTP depth sweep",
        "",
        f"- Date: {state['completed_at_utc']}",
        "- Hardware: NVIDIA GB10",
        f"- Context allocation: {CTX_SIZE} tokens",
        "- KV cache: Q8_0 K/V",
        "- GPU offload: all layers",
        "- Sampling: temperature 0, seed 42",
        "- Matrix: baseline plus MTP2/3/4/5/7/8/9/10/11/12/13/14/15/16; three workloads at ~8K/~32K/~64K prompt lengths",
        "",
        "## Overall",
        "",
        "| Mode | Decode tok/s | Speedup | Prompt tok/s | Acceptance | Quality | Exact | Semantic | Min case tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_name, _ in MODES:
        mode_state = state["modes"][mode_name]
        aggregate = mode_state["aggregate"]
        lines.append(
            f"| {mode_name} | {num(aggregate['decode_tokens_per_second'])} | "
            f"{num(aggregate['speedup_vs_baseline'])}x | "
            f"{num(aggregate['prompt_tokens_per_second'])} | "
            f"{pct(aggregate['draft_acceptance'])} | "
            f"{aggregate['passed']}/{aggregate['total']} | "
            f"{mode_state['exact_matches_with_baseline']}/{aggregate['total']} | "
            f"{mode_state['semantic_matches_with_baseline']}/{aggregate['total']} | "
            f"{num(aggregate['minimum_case_decode_tokens_per_second'])} |"
        )
    lines.extend([
        "", "## By context", "",
        "| Mode | Context | Actual prompt tokens | Decode tok/s | Prompt tok/s | Acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode_name, _ in MODES:
        mode_state = state["modes"][mode_name]
        for context_name, _ in CONTEXT_TARGETS:
            selected = [
                item for item in mode_state["cases"] if item["context"] == context_name
            ]
            actual_prompt = sum(
                item["timings"].get("prompt_n", 0) for item in selected
            ) // max(len(selected), 1)
            aggregate = mode_state["by_context"][context_name]
            lines.append(
                f"| {mode_name} | {context_name} | {actual_prompt} | "
                f"{num(aggregate['decode_tokens_per_second'])} | "
                f"{num(aggregate['prompt_tokens_per_second'])} | "
                f"{pct(aggregate['draft_acceptance'])} |"
            )
    winner = state.get("winner")
    lines.extend(["", "## Decision", ""])
    if winner:
        winner_state = state["modes"][winner]
        depth = winner_state["depth"]
        aggregate = winner_state["aggregate"]
        lines.extend([
            f"- Winner: **MTP{depth}**.",
            f"- Aggregate decode: **{aggregate['decode_tokens_per_second']:.2f} tok/s**.",
            f"- Speedup vs baseline: **{aggregate['speedup_vs_baseline']:.2f}x**.",
            f"- Draft acceptance: **{aggregate['draft_acceptance']:.2%}**.",
            f"- Objective quality: **{aggregate['passed']}/{aggregate['total']}**.",
            f"- Byte-identical with baseline: **{winner_state['exact_matches_with_baseline']}/{aggregate['total']}**.",
            f"- Semantically identical with baseline: **{winner_state['semantic_matches_with_baseline']}/{aggregate['total']}**.",
        ])
    else:
        lines.append(
            "- No MTP depth passed every objective check with byte-identical baseline output."
        )
    lines.extend([
        "",
        "Draft depth is selected only among modes that pass every objective check and remain semantically identical with baseline. Exact byte differences are reported separately.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        state = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "model": "Qwen3.8-27B UD-Q4_K_XL",
            "model_path": str(MODEL),
            "server": str(SERVER),
            "ctx_size": CTX_SIZE,
            "modes": {},
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    cases = None
    total_cases = len(CONTEXT_TARGETS) * len(workload_definitions())
    for mode_name, depth in MODES:
        existing = state["modes"].setdefault(
            mode_name, {"depth": depth, "cases": []}
        )
        completed_ids = {item["id"] for item in existing["cases"]}
        if len(completed_ids) == total_cases:
            print(f"MODE_SKIP {mode_name}: already complete", flush=True)
            continue
        log_path = ROOT / f"{mode_name}_server.log"
        print(f"MODE_START {mode_name} depth={depth}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                server_command(depth),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_ready(process, log_path)
                print(
                    f"SERVER_READY {mode_name} rss_mib={num(rss_mib(process.pid), 1)}",
                    flush=True,
                )
                warm_up()
                if cases is None:
                    print("BUILD_PROMPTS start", flush=True)
                    cases = build_cases()
                    for case in cases:
                        print(
                            f"PROMPT_READY {case['id']} raw_tokens={case['raw_prompt_tokens']}",
                            flush=True,
                        )
                for case in cases:
                    if case["id"] in completed_ids:
                        continue
                    print(f"CASE_START {mode_name} {case['id']}", flush=True)
                    result = run_case(case, process)
                    existing["cases"].append(result)
                    save_state(state)
                    timings = result["timings"]
                    print(
                        f"CASE_DONE {mode_name} {case['id']} "
                        f"prompt_n={timings.get('prompt_n')} "
                        f"pp={num(timings.get('prompt_per_second'))} "
                        f"gen_n={timings.get('predicted_n')} "
                        f"tg={num(timings.get('predicted_per_second'))} "
                        f"accept={timings.get('draft_n_accepted', 0)}/"
                        f"{timings.get('draft_n', 0)} "
                        f"pass={result['score']['passed']} "
                        f"wall={result['wall_seconds']:.1f}s",
                        flush=True,
                    )
            finally:
                stop_server(process)
        print(f"MODE_DONE {mode_name}", flush=True)
    state = finalize(state)
    save_state(state)
    write_report(state)
    summary = {
        mode_name: {
            **state["modes"][mode_name]["aggregate"],
            "exact_matches_with_baseline": state["modes"][mode_name][
                "exact_matches_with_baseline"
            ],
        }
        for mode_name, _ in MODES
    }
    print(
        json.dumps({"winner": state["winner"], "summary": summary}, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
