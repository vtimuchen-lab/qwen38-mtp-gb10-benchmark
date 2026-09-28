"""Paired statistics (moved from make_report.py)."""

from __future__ import annotations

import math
from typing import Any


def exact_binomial_two_sided(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    smaller = min(a, b)
    probability = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return float(min(1.0, 2 * probability))


def paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    """Paired correctness table and exact McNemar p for two id->pass maps."""
    if left.keys() != right.keys():
        raise ValueError("paired key mismatch")
    both = sum(left[item] and right[item] for item in left)
    left_only = sum(left[item] and not right[item] for item in left)
    right_only = sum(not left[item] and right[item] for item in left)
    neither = sum(not left[item] and not right[item] for item in left)
    return {
        "both_correct": both,
        "a_only": left_only,
        "b_only": right_only,
        "both_wrong": neither,
        "discordant": left_only + right_only,
        "mcnemar_exact_p": exact_binomial_two_sided(left_only, right_only),
    }
