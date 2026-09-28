"""Parameter sweeps: one managed server per grid point, suites via ``Runner``.

A sweep file names a base run config and a grid of parameters. Every string
in the base config may contain ``{name}`` placeholders; each grid point
substitutes its values, starts the server from ``[server] command``, waits for
``/health``, runs the suites through the ordinary ``Runner`` and stops the
server. Each point writes a ``sparkbench.result.v1`` file into
``<output_dir>/points/``; a point whose server does not come up (or whose run
fails) is recorded as ``<id>.failed.json`` with the server log and the sweep
moves on. Re-running the sweep skips points that already have a valid,
complete result with the same ``workload_sha256``.

The report ranks points by quality and decode tok/s, marks the Pareto front
and applies the selection rule from ``[select]``.
"""

from __future__ import annotations

import itertools
import re
import shlex
import time
import tomllib
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparkbench import SCHEMA_ID, __version__
from sparkbench.config import ALL_SUITES, Config, ConfigError, parse_config
from sparkbench.jsonutil import read_json, save_json, sha256_json, utc_now

JSONDict = dict[str, Any]

FAILURE_SCHEMA = "sparkbench.sweep_failure.v1"
REPORT_SCHEMA = "sparkbench.sweep_report.v1"
RULES = ("max_speed_within_baseline", "max_speed_min_quality", "max_quality")
QUALITY_METRICS = ("macro", "micro")

_SECTIONS: dict[str, set[str]] = {
    "sweep": {"name", "base_config", "output_dir", "suites", "smoke", "ready_timeout"},
    "grid": set(),
    "exclude": set(),
    "include": set(),
    "select": {"rule", "baseline", "max_quality_drop_pp", "min_quality_pct", "quality"},
    "base": set(),  # an inline run config (validated per point by parse_config)
}
_WHOLE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")
_EPSILON = 1e-9


class SweepError(ConfigError):
    """Malformed sweep file or a base config that cannot be expanded."""


# -- values and substitution ----------------------------------------------


def value_key(value: Any) -> Any:
    """What identifies a grid value: the ``label`` of a table, else the value."""
    return value["label"] if isinstance(value, dict) else value


def same(a: Any, b: Any) -> bool:
    """Equality that keeps ``true`` distinct from ``1`` (TOML types differ)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return bool(a == b)


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", format_value(value)).strip("_") or "_"


def substitute(value: Any, variables: dict[str, Any], _depth: int = 0) -> Any:
    """Replace ``{name}`` placeholders recursively.

    A string that is exactly ``"{name}"`` takes the variable's own TOML type
    (a list is spliced into an enclosing list); placeholders inside longer
    strings are formatted (booleans as ``true``/``false``). ``{{`` and ``}}``
    are literal braces. Unknown names are an error, so typos cannot pass.
    """
    if _depth > 16:
        raise SweepError("placeholders nest too deeply (a variable refers to itself?)")
    if isinstance(value, dict):
        return {key: substitute(item, variables, _depth) for key, item in value.items()}
    if isinstance(value, list):
        result: list[Any] = []
        for item in value:
            whole = _WHOLE.fullmatch(item) if isinstance(item, str) else None
            if whole and isinstance(variables.get(whole[1]), list):
                result.extend(substitute(variables[whole[1]], variables, _depth + 1))
            else:
                result.append(substitute(item, variables, _depth))
        return result
    if not isinstance(value, str):
        return value
    whole = _WHOLE.fullmatch(value)
    if whole:
        return substitute(_lookup(variables, whole[1]), variables, _depth + 1)

    def replace(match: re.Match[str]) -> str:
        if match[0] == "{{":
            return "{"
        if match[0] == "}}":
            return "}"
        found = substitute(_lookup(variables, match[1]), variables, _depth + 1)
        if isinstance(found, (list, dict)):
            raise SweepError(f"{{{match[1]}}} is a list/table and must be a whole string, not part of {value!r}")
        return format_value(found)

    return _PLACEHOLDER.sub(replace, value)


def _lookup(variables: dict[str, Any], name: str) -> Any:
    if name not in variables:
        known = ", ".join(sorted(variables)) or "none"
        raise SweepError(f"unknown placeholder {{{name}}} (known: {known}; write {{{{ }}}} for literal braces)")
    return variables[name]


# -- config -----------------------------------------------------------------


@dataclass(frozen=True)
class SelectRule:
    rule: str = "max_speed_within_baseline"
    baseline: str | dict[str, Any] = "best"
    max_quality_drop_pp: float = 0.0
    min_quality_pct: float | None = None
    quality: str = "macro"

    def describe(self) -> str:
        if self.rule == "max_quality":
            return "highest quality; ties broken by decode tok/s"
        if self.rule == "max_speed_min_quality":
            return f"highest decode tok/s among points with quality >= {self.min_quality_pct:g}%"
        reference = "the best-quality point" if isinstance(self.baseline, str) else f"baseline {_describe_match(self.baseline)}"
        return f"highest decode tok/s among points whose quality is at most {self.max_quality_drop_pp:g} pp below {reference}"


@dataclass(frozen=True)
class Point:
    id: str
    values: dict[str, Any]
    variables: dict[str, Any]


@dataclass(frozen=True)
class SweepConfig:
    path: str
    name: str
    base_path: Path
    base_raw: dict[str, Any]
    output_dir: Path
    suites: tuple[str, ...] | None
    smoke: bool
    ready_timeout: float | None
    axes: dict[str, list[Any]]
    exclude: list[dict[str, Any]]
    include: list[dict[str, Any]]
    select: SelectRule
    raw: dict[str, Any] = field(repr=False)

    @property
    def points_dir(self) -> Path:
        return self.output_dir / "points"


def _describe_match(match: dict[str, Any]) -> str:
    return ", ".join(f"{key}={format_value(value)}" for key, value in match.items())


def _check_axis_value(axis: str, value: Any) -> None:
    if isinstance(value, dict):
        if not isinstance(value.get("label"), str) or not value["label"]:
            raise SweepError(f"[grid] {axis}: a table value needs a non-empty string 'label'")
        return
    if not isinstance(value, (str, int, float, bool)):
        raise SweepError(f"[grid] {axis}: values must be strings, numbers, booleans or tables with a label")


def _table_list(raw: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value = raw.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SweepError(f"[[{name}]] must be an array of tables")
    return list(value)


def parse_sweep(raw: dict[str, Any], path: str = "<memory>") -> SweepConfig:
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise SweepError(f"unknown section(s) in sweep file: {', '.join(unknown)}")
    for name in ("sweep", "grid", "select"):
        if not isinstance(raw.get(name, {}), dict):
            raise SweepError(f"[{name}] must be a table")
    sweep: dict[str, Any] = raw.get("sweep", {})
    select: dict[str, Any] = raw.get("select", {})
    for name, table in (("sweep", sweep), ("select", select)):
        extra = sorted(set(table) - _SECTIONS[name])
        if extra:
            raise SweepError(f"unknown key(s) in [{name}]: {', '.join(extra)}")
    if ("base_config" in sweep) == ("base" in raw):
        raise SweepError("give exactly one of [sweep] base_config (a run config, relative to the sweep file) or an inline [base] table")
    if "base" in raw:
        if not isinstance(raw["base"], dict):
            raise SweepError("[base] must be a table (a run config)")
        base_path = Path(path)
        base_raw: dict[str, Any] = raw["base"]
    else:
        here = Path(path).parent if path != "<memory>" else Path(".")
        base_path = here / str(sweep["base_config"])
        with base_path.open("rb") as handle:
            base_raw = tomllib.load(handle)
    name = str(sweep.get("name") or (Path(path).stem if path != "<memory>" else "sweep"))

    suites: tuple[str, ...] | None = None
    if "suites" in sweep:
        suites = tuple(str(item) for item in sweep["suites"])
        bad = [item for item in suites if item not in ALL_SUITES]
        if bad or not suites:
            raise SweepError(f"[sweep] suites must be a non-empty subset of {', '.join(ALL_SUITES)}")

    axes: dict[str, list[Any]] = {}
    for axis, values in dict(raw.get("grid", {})).items():
        if not _WHOLE.fullmatch("{" + axis + "}"):
            raise SweepError(f"[grid] {axis}: axis names must be identifiers (letters, digits, _)")
        if not isinstance(values, list) or not values:
            raise SweepError(f"[grid] {axis} must be a non-empty array")
        for value in values:
            _check_axis_value(axis, value)
        keys = [value_key(value) for value in values]
        if any(same(a, b) for a, b in itertools.combinations(keys, 2)):
            raise SweepError(f"[grid] {axis}: duplicate values/labels")
        axes[axis] = list(values)

    exclude = _table_list(raw, "exclude")
    include = _table_list(raw, "include")
    for kind, tables in (("exclude", exclude), ("include", include)):
        for table in tables:
            stray = sorted(set(table) - set(axes))
            if stray:
                raise SweepError(f"[[{kind}]] refers to unknown axis/axes: {', '.join(stray)}")
    for table in include:
        missing = sorted(set(axes) - set(table))
        if missing:
            raise SweepError(f"[[include]] must give every axis; missing: {', '.join(missing)}")

    rule = SelectRule(
        rule=str(select.get("rule", "max_speed_within_baseline")),
        baseline=select.get("baseline", "best"),
        max_quality_drop_pp=float(select.get("max_quality_drop_pp", 0.0)),
        min_quality_pct=float(select["min_quality_pct"]) if "min_quality_pct" in select else None,
        quality=str(select.get("quality", "macro")),
    )
    if rule.rule not in RULES:
        raise SweepError(f"[select] rule must be one of {', '.join(RULES)}")
    if rule.quality not in QUALITY_METRICS:
        raise SweepError(f"[select] quality must be one of {', '.join(QUALITY_METRICS)}")
    if rule.rule == "max_speed_min_quality" and rule.min_quality_pct is None:
        raise SweepError("[select] rule max_speed_min_quality needs min_quality_pct")
    if rule.max_quality_drop_pp < 0:
        raise SweepError("[select] max_quality_drop_pp must be >= 0")
    if isinstance(rule.baseline, dict):
        stray = sorted(set(rule.baseline) - set(axes))
        if stray or not rule.baseline:
            raise SweepError(f"[select] baseline must match grid axes; unknown: {', '.join(stray) or '(empty)'}")
    elif rule.baseline != "best":
        raise SweepError('[select] baseline must be "best" or a table of axis values')

    ready_timeout = sweep.get("ready_timeout")
    return SweepConfig(
        path=path,
        name=name,
        base_path=base_path,
        base_raw=base_raw,
        output_dir=Path(str(sweep.get("output_dir", f"results/sweeps/{name}"))),
        suites=suites,
        smoke=bool(sweep.get("smoke", False)),
        ready_timeout=float(ready_timeout) if ready_timeout is not None else None,
        axes=axes,
        exclude=exclude,
        include=include,
        select=rule,
        raw=raw,
    )


def load_sweep(path: str | Path) -> SweepConfig:
    file = Path(path)
    with file.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_sweep(raw, str(file))


# -- points -----------------------------------------------------------------


def _matches(values: dict[str, Any], pattern: dict[str, Any]) -> bool:
    for axis, wanted in pattern.items():
        options = wanted if isinstance(wanted, list) else [wanted]
        if not any(same(value_key(values[axis]), value_key(option)) for option in options):
            return False
    return True


def _resolve_include(sweep: SweepConfig, table: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for axis, choices in sweep.axes.items():
        given = table[axis]
        if isinstance(given, dict):
            _check_axis_value(axis, given)
            values[axis] = given
            continue
        found = [choice for choice in choices if same(value_key(choice), given)]
        if found:
            values[axis] = found[0]
        elif any(isinstance(choice, dict) for choice in choices):
            raise SweepError(f"[[include]] {axis} = {given!r} is not a label in [grid] {axis}; give a full table instead")
        else:
            _check_axis_value(axis, given)
            values[axis] = given
    return values


def point_variables(values: dict[str, Any]) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    for axis, value in values.items():
        names = [axis] + ([key for key in value if key != "label"] if isinstance(value, dict) else [])
        clash = [name for name in names if name in variables]
        if clash:
            raise SweepError(f"variable(s) defined twice across grid axes: {', '.join(clash)}")
        variables[axis] = value_key(value)
        if isinstance(value, dict):
            variables.update({key: item for key, item in value.items() if key != "label"})
    return variables


def expand_points(sweep: SweepConfig) -> list[Point]:
    """Grid product minus ``[[exclude]]``, plus ``[[include]]``, in grid order."""
    combos: list[dict[str, Any]] = []
    names = list(sweep.axes)
    for choice in itertools.product(*(sweep.axes[name] for name in names)):
        values = dict(zip(names, choice, strict=True))
        if not any(_matches(values, pattern) for pattern in sweep.exclude):
            combos.append(values)
    combos += [_resolve_include(sweep, table) for table in sweep.include]
    points: list[Point] = []
    seen: dict[str, dict[str, Any]] = {}
    for values in combos:
        point_id = "__".join(f"{axis}-{_slug(value_key(value))}" for axis, value in values.items()) or "default"
        if point_id in seen:
            if seen[point_id] != values:
                raise SweepError(f"two different points map to the same id {point_id!r}; use distinct labels")
            continue
        seen[point_id] = values
        variables = point_variables(values)
        variables.setdefault("point", point_id)
        points.append(Point(point_id, values, variables))
    if not points:
        raise SweepError("the sweep has no points (empty [grid] and no [[include]], or everything excluded)")
    return points


def point_config(sweep: SweepConfig, point: Point) -> Config:
    """The run config for one point: base config with the point's values."""
    raw = substitute(sweep.base_raw, point.variables)
    run = dict(raw.get("run", {}))
    run["label"] = f"{sweep.name}-{point.id}"
    run["output"] = (sweep.points_dir / f"{point.id}.json").as_posix()
    if sweep.suites is not None:
        run["suites"] = list(sweep.suites)
    raw["run"] = run
    # Grid values go into [parameters], which workload_sha256 covers: the
    # server command itself is host-specific and deliberately not hashed.
    raw["parameters"] = {**dict(raw.get("parameters", {})), "sweep_point": dict(point.values)}
    if "server" in raw:
        server = dict(raw["server"])
        if isinstance(server.get("command"), list):
            # "{block}" keeps its TOML type elsewhere; a command line is text.
            server["command"] = [item if isinstance(item, (list, dict)) else format_value(item) for item in server["command"]]
        server["log"] = (sweep.points_dir / f"{point.id}.server.log").as_posix()
        if sweep.ready_timeout is not None:
            server["ready_timeout"] = sweep.ready_timeout
        raw["server"] = server
    try:
        return parse_config(raw, str(sweep.base_path))
    except ConfigError as exc:
        raise SweepError(f"point {point.id}: {exc}") from exc


# -- state on disk ------------------------------------------------------------


def result_path(sweep: SweepConfig, point: Point) -> Path:
    return sweep.points_dir / f"{point.id}.json"


def failure_path(sweep: SweepConfig, point: Point) -> Path:
    return sweep.points_dir / f"{point.id}.failed.json"


def valid_result(path: Path, config: Config, smoke: bool) -> JSONDict | None:
    """The existing result if it can stand in for running this point again."""
    from sparkbench.schema import semantic_errors, validate

    if not path.exists():
        return None
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != SCHEMA_ID:
        return None
    if document.get("workload_sha256") != config.workload_sha256 or bool(document.get("smoke")) != smoke:
        return None
    if document.get("complete") is not True or not set(config.suites) <= set(document.get("suites") or {}):
        return None
    if validate(document) or semantic_errors(document):
        return None
    return document


def recorded_failure(path: Path, config: Config) -> JSONDict | None:
    if not path.exists():
        return None
    try:
        failure = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(failure, dict) or failure.get("workload_sha256") != config.workload_sha256:
        return None
    return failure


def _tail(path: Path | None, limit: int = 4000) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


# -- execution ----------------------------------------------------------------


@dataclass
class PointOutcome:
    point: Point
    status: str  # "cached" | "done" | "failed"
    path: Path
    stage: str | None = None
    error: str | None = None


def _checkpoint_matches(raw_path: Path, config: Config, smoke: bool) -> bool:
    if not raw_path.exists():
        return False
    try:
        state = read_json(raw_path)
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("config_sha256") == config.config_sha256 and bool(state.get("smoke")) == smoke


def _port_busy(config: Config) -> bool:
    from sparkbench.backend import OpenAIBackend

    if config.server is None:
        return False
    backend = OpenAIBackend(config.backend)
    deadline = time.monotonic() + min(config.server.stop_timeout, 30.0)
    while backend.healthy():
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.5)
    return False


def run_point(
    sweep: SweepConfig,
    point: Point,
    config: Config,
    *,
    smoke: bool,
    fresh: bool,
    log: Callable[[str], None],
) -> PointOutcome:
    from sparkbench.backend import ServerStartError
    from sparkbench.runner import Runner, output_paths
    from sparkbench.schema import semantic_errors, validate

    path = result_path(sweep, point)
    failed = failure_path(sweep, point)
    failed.unlink(missing_ok=True)
    log_path = Path(config.server.log) if config.server is not None and config.server.log else None
    stage = "run"
    try:
        if _port_busy(config):
            raise ServerStartError(
                f"something already answers {config.backend.base_url}{config.backend.health_path} before start; "
                "refusing to benchmark a server this sweep did not launch"
            )
        raw_path = output_paths(config, smoke, str(path))[1]
        runner = Runner(config, smoke=smoke, output=str(path), fresh=fresh or not _checkpoint_matches(raw_path, config, smoke), log=log)
        runner.run()
        document = read_json(path)
        problems = validate(document) or semantic_errors(document)
        if problems or document.get("complete") is not True:
            stage = "invalid_result"
            raise RuntimeError("result is incomplete or invalid: " + "; ".join(problems[:5]))
    except ServerStartError as exc:
        return _fail(sweep, point, config, "server_start", exc, log_path, log)
    except Exception as exc:  # a long unattended sweep records the failure and goes on
        return _fail(sweep, point, config, stage, exc, log_path, log)
    return PointOutcome(point, "done", path)


def _fail(
    sweep: SweepConfig,
    point: Point,
    config: Config,
    stage: str,
    exc: BaseException,
    log_path: Path | None,
    log: Callable[[str], None],
) -> PointOutcome:
    error = f"{type(exc).__name__}: {exc}"
    record = {
        "schema": FAILURE_SCHEMA,
        "sweep": sweep.name,
        "point": point.id,
        "values": point.values,
        "stage": stage,
        "error": error,
        "traceback": traceback.format_exception(exc),
        "server_command": config.server.command if config.server is not None else None,
        "server_log": log_path.as_posix() if log_path is not None else None,
        "server_log_tail": _tail(log_path),
        "workload_sha256": config.workload_sha256,
        "config_sha256": config.config_sha256,
        "failed_at_utc": utc_now(),
        "tool": f"sparkbench {__version__}",
    }
    path = failure_path(sweep, point)
    save_json(path, record)
    log(f"{point.id}: FAILED at {stage}: {error.splitlines()[0]} (see {path})")
    return PointOutcome(point, "failed", path, stage=stage, error=error)


def run_sweep(
    sweep: SweepConfig,
    *,
    smoke: bool | None = None,
    fresh: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[PointOutcome]:
    """Run every point that has no valid result yet; never stops on a failed point."""
    emit = log or (lambda message: print(message, flush=True))
    mode = sweep.smoke if smoke is None else smoke
    points = expand_points(sweep)
    configs = [point_config(sweep, point) for point in points]  # all config errors before any server starts
    sweep.points_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[PointOutcome] = []
    for index, (point, config) in enumerate(zip(points, configs, strict=True), start=1):
        prefix = f"[{index}/{len(points)}] {point.id}"
        if not fresh and valid_result(result_path(sweep, point), config, mode) is not None:
            emit(f"{prefix}: complete result with the same workload_sha256, skipped")
            outcomes.append(PointOutcome(point, "cached", result_path(sweep, point)))
            continue
        emit(f"{prefix}: starting")
        outcome = run_point(sweep, point, config, smoke=mode, fresh=fresh, log=emit)
        if outcome.status == "done":
            emit(f"{prefix}: done")
        outcomes.append(outcome)
    return outcomes


# -- report -------------------------------------------------------------------


@dataclass
class PointRow:
    id: str
    values: dict[str, Any]
    status: str  # "done" | "failed" | "pending"
    result: str | None = None
    quality: float | None = None
    suite_quality: dict[str, float | None] = field(default_factory=dict)
    decode_tps: float | None = None
    draft_acceptance: float | None = None
    pareto: bool = False
    failure: JSONDict | None = None

    @property
    def measured(self) -> bool:
        return self.status == "done" and self.quality is not None and self.decode_tps is not None


def point_quality(document: JSONDict, suites: Iterable[str], metric: str) -> tuple[float | None, dict[str, float | None]]:
    per_suite: dict[str, float | None] = {}
    passed = scored = 0
    for name in suites:
        suite = document["suites"].get(name)
        if suite is None:
            per_suite[name] = None
            continue
        per_suite[name] = suite.get("quality")
        verdicts = suite["verdicts"]
        passed += verdicts["pass"]
        scored += verdicts["pass"] + verdicts["fail"] + verdicts["error"]
    if metric == "micro":
        return (passed / scored if scored else None), per_suite
    values = [value for value in per_suite.values() if value is not None]
    if len(values) != len(per_suite):
        return None, per_suite  # a suite without a score cannot be averaged fairly
    return (sum(values) / len(values) if values else None), per_suite


def pareto_front(rows: list[PointRow]) -> list[str]:
    """Ids not dominated in (quality, decode tok/s); both are maximised."""
    measured = [row for row in rows if row.measured]
    front: list[PointRow] = []
    for row in measured:
        assert row.quality is not None and row.decode_tps is not None
        dominated = False
        for other in measured:
            assert other.quality is not None and other.decode_tps is not None
            if other is row:
                continue
            no_worse = other.quality >= row.quality - _EPSILON and other.decode_tps >= row.decode_tps - _EPSILON
            better = other.quality > row.quality + _EPSILON or other.decode_tps > row.decode_tps + _EPSILON
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    front.sort(key=lambda row: (-(row.decode_tps or 0.0), -(row.quality or 0.0), row.id))
    return [row.id for row in front]


def select_point(rows: list[PointRow], rule: SelectRule) -> JSONDict:
    """Apply the ``[select]`` rule; returns chosen id (or None) with the reasoning."""
    measured = [row for row in rows if row.measured]
    selection: JSONDict = {
        "rule": rule.rule,
        "description": rule.describe(),
        "reference": None,
        "threshold": None,
        "chosen": None,
        "reason": None,
    }
    if not measured:
        selection["reason"] = "no point has both a quality score and a decode tok/s"
        return selection

    def speed_key(row: PointRow) -> tuple[float, float, str]:
        return (-(row.decode_tps or 0.0), -(row.quality or 0.0), row.id)

    if rule.rule == "max_quality":
        best = min(measured, key=lambda row: (-(row.quality or 0.0), -(row.decode_tps or 0.0), row.id))
        selection["chosen"] = best.id
        return selection
    if rule.rule == "max_speed_min_quality":
        assert rule.min_quality_pct is not None
        threshold = rule.min_quality_pct / 100
    else:
        if rule.baseline == "best":
            reference = min(measured, key=lambda row: (-(row.quality or 0.0), -(row.decode_tps or 0.0), row.id))
        else:
            assert isinstance(rule.baseline, dict)
            matching = [row for row in rows if _matches(row.values, rule.baseline)]
            if len(matching) != 1:
                selection["reason"] = f"baseline {_describe_match(rule.baseline)} matches {len(matching)} points, expected exactly 1"
                return selection
            reference = matching[0]
            if not reference.measured:
                selection["reference"] = reference.id
                selection["reason"] = f"baseline point {reference.id} has no measurement (status: {reference.status})"
                return selection
        assert reference.quality is not None
        selection["reference"] = reference.id
        threshold = reference.quality - rule.max_quality_drop_pp / 100
    selection["threshold"] = threshold
    candidates = [row for row in measured if (row.quality or 0.0) >= threshold - _EPSILON]
    if not candidates:
        selection["reason"] = f"no point reaches quality {100 * threshold:.1f}%"
        return selection
    selection["chosen"] = min(candidates, key=speed_key).id
    return selection


def collect_rows(sweep: SweepConfig, smoke: bool | None = None) -> tuple[list[PointRow], tuple[str, ...]]:
    mode = sweep.smoke if smoke is None else smoke
    points = expand_points(sweep)
    rows: list[PointRow] = []
    suites: tuple[str, ...] = ()
    for point in points:
        config = point_config(sweep, point)
        suites = config.suites
        path = result_path(sweep, point)
        document = valid_result(path, config, mode)
        if document is not None:
            quality, per_suite = point_quality(document, config.suites, sweep.select.quality)
            overall = document.get("overall") or {}
            rows.append(
                PointRow(
                    point.id,
                    point.values,
                    "done",
                    result=path.as_posix(),
                    quality=quality,
                    suite_quality=per_suite,
                    decode_tps=overall.get("decode_tokens_per_second"),
                    draft_acceptance=overall.get("draft_acceptance"),
                )
            )
            continue
        failure = recorded_failure(failure_path(sweep, point), config)
        if failure is not None:
            rows.append(PointRow(point.id, point.values, "failed", failure=failure))
        else:
            rows.append(PointRow(point.id, point.values, "pending"))
    front = set(pareto_front(rows))
    for row in rows:
        row.pareto = row.id in front
    return rows, suites


def build_report(sweep: SweepConfig, smoke: bool | None = None) -> JSONDict:
    rows, suites = collect_rows(sweep, smoke)
    selection = select_point(rows, sweep.select)
    return {
        "schema": REPORT_SCHEMA,
        "tool": f"sparkbench {__version__}",
        "generated_at_utc": utc_now(),
        "name": sweep.name,
        "sweep_file": sweep.path,
        "sweep_sha256": sha256_json(sweep.raw),
        "base_config": sweep.base_path.as_posix(),
        "suites": list(suites),
        "smoke": sweep.smoke if smoke is None else smoke,
        "axes": list(sweep.axes),
        "quality_metric": sweep.select.quality,
        "speed_metric": "overall.decode_tokens_per_second",
        "counts": {status: sum(row.status == status for row in rows) for status in ("done", "failed", "pending")},
        "points": [
            {
                "id": row.id,
                "values": row.values,
                "status": row.status,
                "result": row.result,
                "quality": row.quality,
                "suite_quality": row.suite_quality,
                "decode_tokens_per_second": row.decode_tps,
                "draft_acceptance": row.draft_acceptance,
                "pareto": row.pareto,
                "failure": None if row.failure is None else _failure_summary(row.failure),
            }
            for row in rows
        ],
        "pareto_front": pareto_front(rows),
        "selection": selection,
    }


def _failure_summary(failure: JSONDict) -> JSONDict:
    summary = {key: failure.get(key) for key in ("stage", "error", "server_log", "failed_at_utc")}
    lines = [line.strip() for line in str(failure.get("server_log_tail") or "").splitlines() if line.strip()]
    summary["server_log_last_line"] = lines[-1] if lines else None
    return summary


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: JSONDict) -> str:
    points: list[JSONDict] = report["points"]
    by_id = {point["id"]: point for point in points}
    selection: JSONDict = report["selection"]
    reference = by_id.get(selection["reference"]) if selection.get("reference") else None
    has_ref = reference is not None and reference["quality"] is not None and reference["decode_tokens_per_second"]
    axes: list[str] = report["axes"]
    suites: list[str] = report["suites"]
    counts = report["counts"]
    metric = "mean of suite quality scores" if report["quality_metric"] == "macro" else "passed / scored cases over all suites"
    lines = [
        f"# Sweep {report['name']}",
        "",
        f"- Base config: `{report['base_config']}`; suites: {', '.join(suites)}{' (smoke)' if report['smoke'] else ''}.",
        f"- Points: {len(points)} planned, {counts['done']} done, {counts['failed']} failed, {counts['pending']} pending.",
        f"- Quality: {metric}; speed: weighted overall decode tok/s.",
        "",
        "## Points",
        "",
    ]
    header = ["Point", *axes, "Status", "Quality", *suites, "Decode tok/s", "Draft acc."]
    if has_ref:
        header += ["Δ quality (pp)", "Speed vs ref"]
    header.append("Pareto")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---"] * len(axes) + ["---"] + ["---:"] * (len(suites) + 3 + 2 * bool(has_ref)) + [":---:"]) + "|")
    for point in points:
        name = f"**{point['id']}**" if point["id"] == selection.get("chosen") else point["id"]
        cells = [name, *(_cell(format_value(value_key(point["values"][axis]))) for axis in axes), point["status"]]
        cells.append(_pct(point["quality"]))
        cells += [_pct(point["suite_quality"].get(suite)) for suite in suites]
        cells += [_num(point["decode_tokens_per_second"]), _pct(point["draft_acceptance"])]
        if has_ref:
            assert reference is not None
            if point["quality"] is not None:
                cells.append(f"{100 * (point['quality'] - reference['quality']):+.1f}")
            else:
                cells.append("n/a")
            tps = point["decode_tokens_per_second"]
            cells.append(f"{tps / reference['decode_tokens_per_second']:.2f}x" if tps else "n/a")
        cells.append("★" if point["pareto"] else "")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Pareto front (quality × decode tok/s)", ""]
    if report["pareto_front"]:
        lines += ["| Point | Quality | Decode tok/s |", "|---|---:|---:|"]
        for point_id in report["pareto_front"]:
            point = by_id[point_id]
            lines.append(f"| {point_id} | {_pct(point['quality'])} | {_num(point['decode_tokens_per_second'])} |")
    else:
        lines.append("No measured points yet.")

    lines += ["", "## Recommendation", "", f"Rule: {selection['description']}."]
    if reference is not None:
        lines.append(f"Reference: `{reference['id']}` ({_pct(reference['quality'])}, {_num(reference['decode_tokens_per_second'])} tok/s).")
    if selection.get("threshold") is not None:
        lines.append(f"Quality threshold: {_pct(selection['threshold'])}.")
    lines.append("")
    chosen = by_id.get(selection["chosen"]) if selection.get("chosen") else None
    if chosen is not None:
        detail = f"quality {_pct(chosen['quality'])}, {_num(chosen['decode_tokens_per_second'])} tok/s"
        if has_ref:
            assert reference is not None
            detail += (
                f", {100 * (chosen['quality'] - reference['quality']):+.1f} pp and "
                f"{chosen['decode_tokens_per_second'] / reference['decode_tokens_per_second']:.2f}x vs reference"
            )
        values = ", ".join(f"{axis}={format_value(value_key(chosen['values'][axis]))}" for axis in axes)
        lines.append(f"**Recommended: `{chosen['id']}`** ({values}): {detail}.")
    else:
        lines.append(f"No recommendation: {selection['reason']}.")
    if counts["failed"] or counts["pending"]:
        lines.append("")
        lines.append(f"The recommendation covers measured points only ({counts['failed']} failed, {counts['pending']} pending).")

    failed = [point for point in points if point["status"] == "failed"]
    if failed:
        lines += ["", "## Failed points", "", "| Point | Stage | Error | Server log |", "|---|---|---|---|"]
        for point in failed:
            failure = point["failure"] or {}
            error = str(failure.get("error") or "").strip().splitlines()
            first = error[0].rstrip(":")[:200] if error else ""
            last = failure.get("server_log_last_line")
            if last and last not in first:
                first += f" — log: {last[:200]}"
            log = f"`{failure['server_log']}`" if failure.get("server_log") else "n/a"
            lines.append(f"| {point['id']} | {failure.get('stage')} | {_cell(first)} | {log} |")
    return "\n".join(lines) + "\n"


def write_report(sweep: SweepConfig, smoke: bool | None = None) -> tuple[Path, Path, JSONDict]:
    report = build_report(sweep, smoke)
    markdown_path = sweep.output_dir / "report.md"
    json_path = sweep.output_dir / "report.json"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    save_json(json_path, report)
    return markdown_path, json_path, report


def describe_plan(sweep: SweepConfig, smoke: bool | None = None) -> str:
    """Human-readable dry run: every point, its command and whether it would run."""
    mode = sweep.smoke if smoke is None else smoke
    lines = []
    for point in expand_points(sweep):
        config = point_config(sweep, point)
        state = "skip (complete)" if valid_result(result_path(sweep, point), config, mode) else "run"
        lines.append(f"{point.id}: {state}")
        lines.append(f"  workload_sha256 {config.workload_sha256}")
        if config.server is not None:
            lines.append(f"  $ {shlex.join(config.server.command)}")
        if config.backend.extra_body:
            lines.append(f"  extra_body {config.backend.extra_body}")
    return "\n".join(lines) + "\n"
