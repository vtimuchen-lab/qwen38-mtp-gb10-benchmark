"""Command line: ``python -m sparkbench {run,sweep,report,compare-hosts,validate,import}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sparkbench import SCHEMA_ID, __version__
from sparkbench.config import ALL_SUITES, ConfigError, load_config
from sparkbench.jsonutil import read_json, save_json
from sparkbench.schema import semantic_errors, validate


def _load_result(path: str) -> dict[str, object]:
    document = read_json(Path(path))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA_ID:
        raise SystemExit(f"{path}: not a {SCHEMA_ID} file (run `python -m sparkbench import` for legacy files)")
    return document


def _emit(markdown: str, summary: object, out: str | None, json_out: str | None) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(markdown, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)
    if json_out:
        save_json(Path(json_out), summary)
        print(f"wrote {json_out}", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    from sparkbench.backend import BackendError
    from sparkbench.runner import RunError, Runner

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        runner = Runner(config, suites=args.suite or None, smoke=args.smoke, output=args.output, fresh=args.fresh)
        path = runner.run()
    except (RunError, BackendError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = validate(read_json(path))
    print(f"wrote {path} ({'valid' if not errors else f'{len(errors)} schema errors'})")
    return 0 if not errors else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    from sparkbench.sweep import describe_plan, load_sweep, run_sweep, write_report

    smoke = True if args.smoke else None
    try:
        sweep = load_sweep(args.config)
        if args.dry_run:
            sys.stdout.write(describe_plan(sweep, smoke))
            return 0
        outcomes = [] if args.report_only else run_sweep(sweep, smoke=smoke, fresh=args.fresh)
        markdown, summary, report = write_report(sweep, smoke)
    except (ConfigError, OSError) as exc:
        print(f"sweep config error: {exc}", file=sys.stderr)
        return 2
    counts = report["counts"]
    chosen = report["selection"]["chosen"]
    print(f"wrote {markdown} and {summary}: {counts['done']} done, {counts['failed']} failed, {counts['pending']} pending")
    print(f"recommended: {chosen}" if chosen else f"no recommendation: {report['selection']['reason']}")
    failed_now = any(outcome.status == "failed" for outcome in outcomes)
    return 1 if failed_now or (args.report_only and counts["failed"]) else 0


def cmd_report(args: argparse.Namespace) -> int:
    from sparkbench.report import report

    summary, markdown = report(_load_result(args.a), _load_result(args.b))
    _emit(markdown, summary, args.out, args.json)
    return 0


def cmd_compare_hosts(args: argparse.Namespace) -> int:
    from sparkbench.compare_hosts import compare, render_markdown

    result = compare(_load_result(args.a), _load_result(args.b))
    _emit(render_markdown(result), result, args.out, args.json)
    overall = result["overall"]
    if args.fail_on_diff and (overall["numeric_differences"] or overall["text_differences"]):
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    status = 0
    for path in args.files:
        try:
            document = read_json(Path(path))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path}: cannot read JSON: {exc}")
            status = 1
            continue
        errors = validate(document)
        if not errors:
            errors = semantic_errors(document)
        if errors:
            status = 1
            print(f"{path}: INVALID ({len(errors)} error(s))")
            for error in errors[: args.max_errors]:
                print(f"  {error}")
        else:
            print(f"{path}: valid {SCHEMA_ID}")
    return status


def cmd_import(args: argparse.Namespace) -> int:
    from sparkbench.importers import import_all

    written = import_all(Path(args.root), Path(args.out_dir))
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sparkbench", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sparkbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run suites against an OpenAI-compatible server")
    run.add_argument("--config", required=True, help="TOML config (see examples/)")
    run.add_argument("--suite", action="append", choices=ALL_SUITES, help="suite to run (repeatable; default: config [run] suites)")
    run.add_argument("--smoke", action="store_true", help="2-3 cases per suite; writes <output>_smoke.json")
    run.add_argument("--output", help="override [run] output")
    run.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint instead of resuming")
    run.set_defaults(func=cmd_run)

    sweep = sub.add_parser("sweep", help="run a parameter grid (one managed server per point) and rank the points")
    sweep.add_argument("--config", required=True, help="sweep TOML (see examples/sweep-qwen38-sglang.toml)")
    sweep.add_argument("--smoke", action="store_true", help="2-3 cases per suite (overrides [sweep] smoke)")
    sweep.add_argument("--fresh", action="store_true", help="re-run every point, ignoring existing results")
    sweep.add_argument("--dry-run", action="store_true", help="print the points and server commands, run nothing")
    sweep.add_argument("--report-only", action="store_true", help="rebuild report.md/report.json from existing results")
    sweep.set_defaults(func=cmd_sweep)

    report = sub.add_parser("report", help="paired A/B report (quality, tok/s, deltas, McNemar)")
    report.add_argument("a")
    report.add_argument("b")
    report.add_argument("--out", help="write Markdown here instead of stdout")
    report.add_argument("--json", help="also write the summary JSON here")
    report.set_defaults(func=cmd_report)

    hosts = sub.add_parser("compare-hosts", help="same config on two hosts: tok/s deltas and answer identity")
    hosts.add_argument("a")
    hosts.add_argument("b")
    hosts.add_argument("--out", help="write Markdown here instead of stdout")
    hosts.add_argument("--json", help="also write the comparison JSON here")
    hosts.add_argument("--fail-on-diff", action="store_true", help="exit 1 if any paired answer differs")
    hosts.set_defaults(func=cmd_compare_hosts)

    check = sub.add_parser("validate", help=f"check files against schemas/result.v1.json ({SCHEMA_ID})")
    check.add_argument("files", nargs="+")
    check.add_argument("--max-errors", type=int, default=20)
    check.set_defaults(func=cmd_validate)

    imp = sub.add_parser("import", help="convert historical result files into results/v1/")
    imp.add_argument("--root", default=".", help="repository root containing the historical files")
    imp.add_argument("--out-dir", default="results/v1")
    imp.set_defaults(func=cmd_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code
