"""Fake OpenAI-compatible server with canned, deterministic answers.

Used in-process (``FakeServer`` in a thread) and as a subprocess
(``python tests/fakeserver.py --port N``) to test the managed-server path.

``flavor="llama.cpp"`` adds llama.cpp ``timings``; ``flavor="vllm"`` returns
plain OpenAI responses (usage only). ``speed`` scales the reported decode
rate and ``variant`` switches a few answers to simulate another host
(``"sloppy"`` breaks the stability answer, to simulate a lossy setting).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkbench.suites.tools import tool_cases

TOOL_ANSWERS = {row["prompt"]: row for row in tool_cases()}


def fake_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def answer(messages: list[dict[str, Any]], payload: dict[str, Any], variant: str) -> tuple[str, list[dict[str, Any]] | None]:
    """Return (content, tool_calls) for a request."""
    last = str(messages[-1].get("content", ""))
    if payload.get("tools"):
        row = TOOL_ANSWERS.get(last)
        if row is None:
            return "I cannot help with that.", None
        args = dict(row["args"])
        if row["name"] == "lookup_order" and row["args"]["order_id"] == "ZX-900":
            args = {"order_id": "ZX900"}  # one deliberate miss
        call = {"type": "function", "id": "call_1", "function": {"name": row["name"], "arguments": json.dumps(args)}}
        return "", [call]
    if last == "Reply with OK.":
        return "OK", None
    if last.startswith("Now give only the final choice"):
        previous = str(messages[-2].get("content", ""))
        match = re.search(r"Answer:\s*([A-J])", previous)
        return f"Answer: {match.group(1) if match else 'A'}", None
    if "FINAL PROBLEM:" in last and "multiple-choice" in last:
        if "7 * 8" in last:
            return "7 times 8 is 56.\nAnswer: B", None
        if "atmosphere" in last:
            return "Nitrogen dominates.\nAnswer: B", None
        return "The derivative of x^2 is x.\nAnswer: A", None  # wrong (expected C)
    if "grade-school math" in last:
        final = last.rsplit("FINAL PROBLEM:", 1)[1]
        if "Tom has 3 apples" in final:
            return ("3 + 4 = 7\n#### 7" if variant != "other" else "3 + 4 = 8\n#### 8"), None
        if "A book costs" in final:
            return "2 * 12 + 3 = 27\n#### 27", None
        return "1200 + 800 = 2000 meters\n#### 2000", None
    if "Complete the Python function below" in last:
        return "return x", None
    if "ACTIVE_NEEDLE" in last:
        secret = re.search(r"ACTIVE_NEEDLE secret=(\S+) owner=(\S+)", last)
        assert secret is not None
        if "strict JSON" in last:
            return json.dumps({"secret": secret.group(1), "owner": secret.group(2)}), None
        if "get_secret" in last:
            return f"def get_secret():\n    return {secret.group(1)!r}", None
        return secret.group(1), None
    if last.startswith("Return exactly this text and nothing else"):
        return ("BENCHMARK_OK_42" if variant != "sloppy" else "BENCHMARK OK 42"), None
    if "lowercase" in last:
        text = "rain taps softly on the roof. the street shines under grey light."
        return (text if variant != "other" else text + " it keeps falling."), None
    if "postscript" in last:
        return "Hi Sam,\n\nIt was great to see you last week. Let's meet again soon.\n\nBest,\nAna", None
    if "lighthouse" in last:
        return " ".join(["The lighthouse stands tall above the rocky shore and guides every ship home."] * 4), None
    return "I am a fake model.", None


class FakeState:
    def __init__(self, flavor: str = "llama.cpp", speed: float = 1.0, variant: str = "base") -> None:
        self.flavor = flavor
        self.speed = speed
        self.variant = variant
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()


def make_handler(state: FakeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/props" and state.flavor == "llama.cpp":
                self._send(200, {"build_info": "b1-fake"})
            elif self.path == "/version" and state.flavor == "vllm":
                self._send(200, {"version": "0.0-fake"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            with state.lock:
                state.requests.append({"path": self.path, "payload": payload})
            if self.path == "/tokenize":
                text = payload.get("content", payload.get("prompt", ""))
                count = fake_token_count(text)
                body: dict[str, Any] = {"tokens": [0] * count}
                if state.flavor == "vllm":
                    body["count"] = count
                self._send(200, body)
                return
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            messages = payload["messages"]
            content, tool_calls = answer(messages, payload, state.variant)
            prompt_n = sum(fake_token_count(str(m.get("content", ""))) for m in messages)
            predicted_n = max(1, len(content) // 4) if content else 12
            predicted_n = min(predicted_n, int(payload.get("max_tokens", 10**9)))
            decode_rate = 30.0 * state.speed
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            response: dict[str, Any] = {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": payload.get("model"),
                "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
                "usage": {"prompt_tokens": prompt_n, "completion_tokens": predicted_n, "total_tokens": prompt_n + predicted_n},
            }
            if state.flavor == "llama.cpp":
                predicted_ms = predicted_n / decode_rate * 1000
                prompt_ms = prompt_n / 600.0 * 1000
                draft_n = predicted_n + 2
                response["timings"] = {
                    "cache_n": 0,
                    "prompt_n": prompt_n,
                    "prompt_ms": prompt_ms,
                    "prompt_per_second": prompt_n / (prompt_ms / 1000),
                    "predicted_n": predicted_n,
                    "predicted_ms": predicted_ms,
                    "predicted_per_second": predicted_n / (predicted_ms / 1000),
                    "draft_n": draft_n,
                    "draft_n_accepted": predicted_n - 1,
                }
            self._send(200, response)

    return Handler


class FakeServer:
    def __init__(self, flavor: str = "llama.cpp", speed: float = 1.0, variant: str = "base", port: int = 0) -> None:
        self.state = FakeState(flavor, speed, variant)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.state))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--flavor", default="llama.cpp")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0, help="multiplied into --speed (a second sweep axis)")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--crash", action="store_true", help="print an error and exit 3 instead of serving")
    parser.add_argument("--hang", action="store_true", help="never open the port (health never passes)")
    args = parser.parse_args()
    if args.crash:
        print("fake: CUDA out of memory while allocating KV cache", file=sys.stderr, flush=True)
        raise SystemExit(3)
    if args.hang:
        import contextlib
        import time

        with contextlib.suppress(KeyboardInterrupt):
            time.sleep(3600)
        return
    server = FakeServer(args.flavor, speed=args.speed * args.scale, variant=args.variant, port=args.port)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()


if __name__ == "__main__":
    main()
