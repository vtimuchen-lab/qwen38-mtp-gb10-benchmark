#!/usr/bin/env python3
"""Download and freeze deterministic benchmark subsets from primary datasets."""

from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 42


def get_json(url: str, retries: int = 8) -> dict:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "qwen38-quality-eval/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network retries are intentionally broad
            error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"failed to fetch {url}: {error}")


def fetch_rows(dataset: str, config: str, split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        payload = get_json(f"https://datasets-server.huggingface.co/rows?{query}")
        if total is None:
            total = int(payload["num_rows_total"])
        page = [item["row"] for item in payload["rows"]]
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        print(f"{dataset}/{split}: {len(rows)}/{total}", flush=True)
    if total is None or len(rows) != total:
        raise RuntimeError(f"incomplete {dataset}/{split}: {len(rows)}/{total}")
    return rows


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stratified_mmlu(rows: list[dict], count: int) -> list[dict]:
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    categories = sorted(groups)
    for category in categories:
        rng.shuffle(groups[category])
    base, extra = divmod(count, len(categories))
    selected: list[dict] = []
    for index, category in enumerate(categories):
        selected.extend(groups[category][: base + (index < extra)])
    rng.shuffle(selected)
    return selected


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    mmlu_test = fetch_rows("TIGER-Lab/MMLU-Pro", "default", "test")
    mmlu_validation = fetch_rows("TIGER-Lab/MMLU-Pro", "default", "validation")
    gsm_test = fetch_rows("openai/gsm8k", "main", "test")
    gsm_train = fetch_rows("openai/gsm8k", "main", "train")

    rng = random.Random(SEED)
    gsm_test_sample = rng.sample(gsm_test, 100)
    gsm_train_shots = rng.sample(gsm_train, 8)

    ifeval_path = (
        ROOT
        / "vendor/google-research/instruction_following_eval/data/input_data.jsonl"
    )
    ifeval_rows = [
        json.loads(line)
        for line in ifeval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ifeval_sample = random.Random(SEED).sample(ifeval_rows, 50)

    write_json(DATA / "mmlu_pro_test_full.json", mmlu_test)
    write_json(DATA / "mmlu_pro_validation.json", mmlu_validation)
    write_json(DATA / "mmlu_pro_test_stratified_100.json", stratified_mmlu(mmlu_test, 100))
    write_json(DATA / "gsm8k_test_sample_100.json", gsm_test_sample)
    write_json(DATA / "gsm8k_train_shots_8.json", gsm_train_shots)
    write_json(DATA / "ifeval_sample_50.json", ifeval_sample)
    write_json(
        DATA / "manifest.json",
        {
            "seed": SEED,
            "mmlu_pro": {
                "source": "TIGER-Lab/MMLU-Pro",
                "revision": "datasets-server snapshot fetched 2026-08-17",
                "test_total": len(mmlu_test),
                "validation_total": len(mmlu_validation),
                "sample": 100,
                "sampling": "category-stratified",
            },
            "gsm8k": {
                "source": "openai/gsm8k",
                "config": "main",
                "test_total": len(gsm_test),
                "train_total": len(gsm_train),
                "sample": 100,
                "shots": 8,
            },
            "humaneval": {
                "source": "openai/human-eval",
                "revision": "cloned 2026-08-17",
                "sample": 40,
            },
            "ifeval": {
                "source": "google-research/instruction_following_eval",
                "revision": "cloned 2026-08-17",
                "total": len(ifeval_rows),
                "sample": 50,
            },
        },
    )


if __name__ == "__main__":
    main()
