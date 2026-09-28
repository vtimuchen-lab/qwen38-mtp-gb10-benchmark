"""Converters from historical result files to ``sparkbench.result.v1``.

Sources are only read. ``import_all`` writes one v1 file per run/mode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sparkbench.importers.extended import import_complex, import_prefix_cache
from sparkbench.importers.legacy_main import import_legacy_main
from sparkbench.importers.soak import import_soak
from sparkbench.importers.sweeps import import_mtp_sweep, import_q4_q6_compare
from sparkbench.jsonutil import save_json
from sparkbench.schema import validate
from sparkbench.suites.common import JSONDict


def _single(name: str, function: Callable[[Path, Path], JSONDict]) -> Callable[[Path, Path], dict[str, JSONDict]]:
    def wrapped(path: Path, root: Path) -> dict[str, JSONDict]:
        return {name: function(path, root)}

    return wrapped


# (source path relative to repo root, importer) in output order.
SOURCES: list[tuple[str, Callable[[Path, Path], dict[str, JSONDict]]]] = [
    ("results/q4.json", _single("main_q4_mtp7", import_legacy_main)),
    ("results/q6.json", _single("main_q6_mtp7", import_legacy_main)),
    ("mtp_sweep/results.json", import_mtp_sweep),
    ("q4_q6_compare/results.json", import_q4_q6_compare),
    ("extended_validation/complex_results.json", import_complex),
    ("extended_validation/prefix_cache_results.json", import_prefix_cache),
    ("soak_partial/summary.json", import_soak),
]


def import_all(root: Path, out_dir: Path) -> dict[str, Path]:
    """Convert every known source under ``root``; returns name -> written path."""
    written: dict[str, Path] = {}
    for relative, importer in SOURCES:
        source = root / relative
        if not source.exists():
            continue
        for name, document in importer(source, root).items():
            errors = validate(document)
            if errors:
                raise ValueError(f"{name}: generated document fails schema: {errors[:5]}")
            target = out_dir / f"{name}.json"
            save_json(target, document)
            written[name] = target
    return written


__all__ = ["SOURCES", "import_all"]
