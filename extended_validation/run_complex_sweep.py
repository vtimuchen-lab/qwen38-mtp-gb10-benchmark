#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path('/home/admin/qwen38_extended_validation_20260819')
QUALITY_ROOT = Path('/home/admin/qwen38_quality_eval_20260817')
SWEEP_RESULTS = Path('/home/admin/qwen38_mtp_sweep_20260817/results.json')
SERVER = Path('/home/admin/llama.cpp-main-20260810/build/bin/llama-server')
MODEL = Path('/usr/share/ollama/.ollama/models/blobs/sha256-bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372')
PORT = 18084
SEED = 42
CTX_SIZE = 65536

spec = importlib.util.spec_from_file_location('quality_benchmark', QUALITY_ROOT / 'run_benchmark.py')
qb = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qb)

CHAT_PROMPTS = [
    'Explain in exactly three short bullet points why deterministic tests need a fixed seed.',
    'A service processed 120 requests: 114 succeeded, 4 timed out, and 2 failed validation. Give success rate and one concise reliability observation.',
    'Write a PostgreSQL query that returns the five customers with the largest paid invoice total in 2026. Tables: customers(id,name), invoices(customer_id,status,paid_at,total). Return SQL only.',
    'Ответь двумя предложениями: чем prefix cache отличается от KV cache?',
    'Find the logical flaw: All fast systems use caching. This system uses caching. Therefore it is fast. Explain briefly.',
    'Convert UTC 2026-08-19 17:45 to UTC+03:00. Return only ISO-8601 local time with offset.',
    'Design a minimal JSON object for a benchmark result with model, tokens_per_second, passed, and timestamp. Return JSON only.',
    'A Python loop mutates a list while iterating over it. State the common failure mode and the safest one-line iteration fix.',
    'Summarize this policy in one sentence: retries are allowed only for transport errors, never for failed assertions, and all retries must be recorded.',
    'Classify the sentiment as positive, neutral, or negative and justify in five words or fewer: The update works, although startup is slower.',
]

def post_json(path: str, payload: dict, timeout: float = 1800.0) -> dict:
    request = urllib.request.Request(
        f'http://127.0.0.1:{PORT}{path}',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc


def save_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def server_command(depth: int | None) -> list[str]:
    command = [
        str(SERVER), '--model', str(MODEL), '--alias', 'qwen3.8-MTP:27b',
        '--host', '127.0.0.1', '--port', str(PORT), '--no-webui', '--offline',
        '--ctx-size', str(CTX_SIZE), '--parallel', '1', '--gpu-layers', 'all',
        '--fit', 'off', '--flash-attn', 'on', '--cache-type-k', 'q8_0',
        '--cache-type-v', 'q8_0', '--batch-size', '2048', '--ubatch-size', '2048',
        '--jinja', '--reasoning', 'off', '--reasoning-budget', '0',
        '--reasoning-format', 'deepseek', '--log-verbosity', '2',
    ]
    if depth is not None:
        command += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(depth), '--spec-draft-ngl', 'all']
    return command


def wait_ready(process: subprocess.Popen, log_path: Path) -> float:
    started = time.monotonic()
    deadline = started + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(log_path.read_text(encoding='utf-8', errors='replace')[-16000:])
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError('llama-server readiness timeout')


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def memory_snapshot(pid: int) -> dict:
    out: dict[str, Any] = {'pid': pid}
    for filename, keys in (
        ('status', {'VmRSS', 'VmHWM', 'VmSize'}),
        ('smaps_rollup', {'Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Shared_Clean', 'Shared_Dirty'}),
    ):
        values = {}
        try:
            text = Path(f'/proc/{pid}/{filename}').read_text(encoding='utf-8', errors='replace')
            for line in text.splitlines():
                name = line.split(':', 1)[0]
                if name in keys:
                    match = re.search(r'(\d+)\s+kB', line)
                    if match:
                        values[name + '_mib'] = int(match.group(1)) / 1024
        except OSError as exc:
            values['error'] = str(exc)
        out[filename] = values
    try:
        meminfo = Path('/proc/meminfo').read_text(encoding='utf-8')
        match = re.search(r'^MemAvailable:\s+(\d+)\s+kB$', meminfo, re.MULTILINE)
        out['system_mem_available_mib'] = int(match.group(1)) / 1024 if match else None
    except OSError:
        out['system_mem_available_mib'] = None
    return out


def compact(response: dict, wall: float) -> dict:
    choice = response.get('choices', [{}])[0]
    message = choice.get('message', {})
    return {
        'content': message.get('content') or message.get('reasoning_content') or '',
        'message': message,
        'finish_reason': choice.get('finish_reason'),
        'usage': response.get('usage', {}),
        'timings': response.get('timings', {}),
        'wall_seconds': wall,
    }


def chat(messages: list[dict], max_tokens: int, **extra: Any) -> dict:
    payload = {
        'model': 'qwen3.8-MTP:27b', 'messages': messages, 'temperature': 0,
        'seed': SEED, 'max_tokens': max_tokens, 'stream': False, 'cache_prompt': False,
    }
    payload.update(extra)
    started = time.monotonic()
    response = post_json('/v1/chat/completions', payload)
    return compact(response, time.monotonic() - started)


def aggregate(cases: list[dict]) -> dict:
    timings = [case['response'].get('timings', {}) for case in cases]
    gen_n = sum(int(item.get('predicted_n', 0) or 0) for item in timings)
    gen_ms = sum(float(item.get('predicted_ms', 0) or 0) for item in timings)
    prompt_n = sum(int(item.get('prompt_n', 0) or 0) for item in timings)
    prompt_ms = sum(float(item.get('prompt_ms', 0) or 0) for item in timings)
    draft_n = sum(int(item.get('draft_n', 0) or 0) for item in timings)
    accepted = sum(int(item.get('draft_n_accepted', 0) or 0) for item in timings)
    return {
        'cases': len(cases), 'prompt_tokens': prompt_n,
        'prompt_tokens_per_second': prompt_n / (prompt_ms / 1000) if prompt_ms else None,
        'generated_tokens': gen_n,
        'decode_tokens_per_second': gen_n / (gen_ms / 1000) if gen_ms else None,
        'draft_generated': draft_n, 'draft_accepted': accepted,
        'draft_acceptance': accepted / draft_n if draft_n else None,
        'wall_seconds': sum(case['response']['wall_seconds'] for case in cases),
    }


def run_mode(mode: str, depth: int | None, state: dict) -> None:
    mode_state = state['modes'].setdefault(mode, {'depth': depth, 'suites': {}})
    if mode_state.get('complete'):
        print(f'MODE_SKIP {mode}', flush=True)
        return
    log_path = ROOT / f'{mode}_complex_server.log'
    print(f'MODE_START {mode}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(server_command(depth), stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            mode_state['load_seconds'] = wait_ready(process, log_path)
            mode_state['memory_ready'] = memory_snapshot(process.pid)
            warm = chat([{'role': 'user', 'content': 'Reply with exactly READY'}], 16)
            mode_state['warmup'] = warm
            time.sleep(1)
            mode_state['memory_after_warmup'] = memory_snapshot(process.pid)
            if depth is not None and int(warm.get('timings', {}).get('draft_n', 0) or 0) <= 0:
                raise RuntimeError(f'{mode}: MTP guard failed: draft_n=0')

            rows = qb.read_json(QUALITY_ROOT / 'data/ifeval_sample_50.json')[:10]
            suite = mode_state['suites'].setdefault('ifeval', {'cases': []})
            done = {case['id'] for case in suite['cases']}
            for index, row in enumerate(rows, 1):
                cid = str(row['key'])
                if cid in done:
                    continue
                result = chat([{'role': 'user', 'content': row['prompt']}], qb.ifeval_max_tokens(row))
                suite['cases'].append({'id': cid, 'prompt': row['prompt'], 'response': result})
                save_json(ROOT / 'complex_results.json', state)
                print(f'{mode} IFEval {index}/10', flush=True)
            suite['scores'] = qb.score_ifeval(suite['cases'], rows)
            suite['aggregate'] = aggregate(suite['cases'])

            test = qb.read_json(QUALITY_ROOT / 'data/gsm8k_test_sample_100.json')[:10]
            shots = qb.read_json(QUALITY_ROOT / 'data/gsm8k_train_shots_8.json')
            demos = '\n\n'.join(f"Question: {item['question']}\nSolution: {item['answer']}" for item in shots)
            suite = mode_state['suites'].setdefault('gsm8k', {'cases': []})
            done = {case['id'] for case in suite['cases']}
            for index, row in enumerate(test, 1):
                cid = hashlib.sha256(row['question'].encode()).hexdigest()[:16]
                if cid in done:
                    continue
                prompt = ('Solve the final grade-school math problem step by step. End with the exact marker `#### number`.\n\n' + demos + f"\n\nFINAL PROBLEM:\nQuestion: {row['question']}\nSolution:")
                result = chat([{'role': 'user', 'content': prompt}], 512)
                predicted = qb.gsm_answer(result['content'])
                expected = qb.gsm_answer(row['answer'])
                suite['cases'].append({'id': cid, 'expected': expected, 'predicted': predicted, 'correct': predicted == expected, 'response': result})
                save_json(ROOT / 'complex_results.json', state)
                print(f'{mode} GSM8K {index}/10 correct={predicted == expected}', flush=True)
            suite['score'] = sum(case['correct'] for case in suite['cases']) / len(suite['cases'])
            suite['aggregate'] = aggregate(suite['cases'])

            tool_rows = qb.tool_cases()[:10]
            suite = mode_state['suites'].setdefault('tools', {'cases': []})
            done = {case['id'] for case in suite['cases']}
            for index, row in enumerate(tool_rows, 1):
                cid = hashlib.sha256(row['prompt'].encode()).hexdigest()[:16]
                if cid in done:
                    continue
                result = chat([{'role': 'user', 'content': row['prompt']}], 256, tools=qb.TOOLS, tool_choice='auto')
                name, arguments = qb.normalized_tool_call(result['message'])
                passed = name == row['name'] and arguments == row['args']
                suite['cases'].append({'id': cid, 'expected': row, 'name': name, 'arguments': arguments, 'passed': passed, 'response': result})
                save_json(ROOT / 'complex_results.json', state)
                print(f'{mode} tools {index}/10 passed={passed}', flush=True)
            suite['score'] = sum(case['passed'] for case in suite['cases']) / len(suite['cases'])
            suite['aggregate'] = aggregate(suite['cases'])

            suite = mode_state['suites'].setdefault('chat', {'cases': []})
            done = {case['id'] for case in suite['cases']}
            for index, prompt in enumerate(CHAT_PROMPTS, 1):
                cid = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                if cid in done:
                    continue
                result = chat([{'role': 'user', 'content': prompt}], 384)
                content = result['content'].strip()
                pathological = bool(re.fullmatch(r'([!?.])\1{7,}', content))
                valid = bool(content) and not pathological
                suite['cases'].append({'id': cid, 'prompt': prompt, 'valid': valid, 'response': result})
                save_json(ROOT / 'complex_results.json', state)
                print(f'{mode} chat {index}/10 valid={valid}', flush=True)
            suite['score'] = sum(case['valid'] for case in suite['cases']) / len(suite['cases'])
            suite['aggregate'] = aggregate(suite['cases'])
            mode_state['memory_after_suite'] = memory_snapshot(process.pid)
            mode_state['complete'] = True
            save_json(ROOT / 'complex_results.json', state)
        finally:
            stop_server(process)
    print(f'MODE_DONE {mode}', flush=True)


def write_report(state: dict) -> None:
    lines = [
        '# Qwen3.8 Q4 complex MTP depth validation', '',
        '- Same frozen 10-case subsets for each mode: IFEval, GSM8K, tool calls, and chat.',
        '- Sampling: temperature 0, seed 42; context 65,536; Q8_0 KV; prefix cache disabled.', '',
        '| Mode | IFEval strict prompt | GSM8K | Tools | Chat valid | Decode tok/s | Acceptance | Warm RSS MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for mode, data in state['modes'].items():
        if not data.get('complete'):
            continue
        suites = data['suites']
        all_cases = sum((suite['cases'] for suite in suites.values()), [])
        overall = aggregate(all_cases)
        data['aggregate'] = overall
        rss = data['memory_after_warmup']['status'].get('VmRSS_mib')
        accept = overall['draft_acceptance']
        lines.append(
            f"| {mode} | {suites['ifeval']['scores']['strict']['prompt_level']:.0%} | "
            f"{suites['gsm8k']['score']:.0%} | {suites['tools']['score']:.0%} | {suites['chat']['score']:.0%} | "
            f"{overall['decode_tokens_per_second']:.2f} | {accept:.2%} | {rss:.1f} |" if accept is not None else
            f"| {mode} | {suites['ifeval']['scores']['strict']['prompt_level']:.0%} | "
            f"{suites['gsm8k']['score']:.0%} | {suites['tools']['score']:.0%} | {suites['chat']['score']:.0%} | "
            f"{overall['decode_tokens_per_second']:.2f} | — | {rss:.1f} |"
        )
    lines += ['', 'The mini-sweep is a paired regression check, not a replacement for the full 50/100-case benchmark.', '']
    (ROOT / 'COMPLEX_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(ROOT / 'complex_results.json', state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', required=True, help='comma-separated: baseline,mtp5,mtp7,mtpN')
    args = parser.parse_args()
    requested = []
    for name in args.modes.split(','):
        name = name.strip()
        if name == 'baseline':
            requested.append((name, None))
        elif re.fullmatch(r'mtp\d+', name):
            requested.append((name, int(name[3:])))
        else:
            raise SystemExit(f'invalid mode: {name}')
    result_path = ROOT / 'complex_results.json'
    state = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': 'Qwen3.8-27B UD-Q4_K_XL', 'modes': {},
    }
    for mode, depth in requested:
        run_mode(mode, depth, state)
        write_report(state)
    state['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_report(state)

if __name__ == '__main__':
    main()
