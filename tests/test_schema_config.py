"""Schema validator, config parsing and config hashes."""

from __future__ import annotations

import copy

import pytest
from conftest import ROOT

from sparkbench.config import ConfigError, load_config, parse_config
from sparkbench.jsonutil import read_json
from sparkbench.schema import SchemaError, Validator, load_schema, semantic_errors, validate

V1 = ROOT / "results" / "v1"


@pytest.fixture(scope="module")
def document() -> dict:
    return read_json(V1 / "main_q4_mtp7.json")


def test_all_committed_v1_files_valid() -> None:
    files = sorted(V1.glob("*.json"))
    assert files
    for path in files:
        doc = read_json(path)
        assert validate(doc) == [], path.name
        assert semantic_errors(doc) == [], path.name


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema="sparkbench.result.v0"), "expected constant"),
        (lambda d: d.pop("config_sha256"), "missing required property 'config_sha256'"),
        (lambda d: d.update(config_sha256="xyz"), "does not match"),
        (lambda d: d["host"].pop("driver"), "missing required property 'driver'"),
        (lambda d: d.update(unexpected=1), "unexpected property 'unexpected'"),
        (lambda d: d["suites"]["tools"]["cases"][0].update(verdict="maybe"), "not in"),
        (lambda d: d["suites"]["tools"]["cases"][0]["tokens"].update(prompt=-1), "minimum"),
        (lambda d: d["suites"]["tools"]["cases"][0]["tok_s"].update(decode="fast"), "expected number/null"),
        (lambda d: d["suites"]["tools"].update(quality=1.5), "maximum"),
        (lambda d: d["suites"]["tools"]["aggregate"].pop("draft_acceptance"), "draft_acceptance"),
    ],
)
def test_schema_rejects(document: dict, mutate: object, message: str) -> None:
    broken = copy.deepcopy(document)
    mutate(broken)  # type: ignore[operator]
    errors = validate(broken)
    assert errors and any(message in error for error in errors), errors


def test_semantic_checks(document: dict) -> None:
    broken = copy.deepcopy(document)
    broken["suites"]["tools"]["cases"][0]["content"] = "tampered"
    broken["suites"]["tools"]["cases"][1]["id"] = broken["suites"]["tools"]["cases"][0]["id"]
    errors = semantic_errors(broken)
    assert any("content_sha256" in e for e in errors) and any("duplicate id" in e for e in errors)


def test_validator_refuses_unknown_keywords() -> None:
    with pytest.raises(SchemaError):
        Validator({"type": "object", "patternProperties": {}}).errors({})
    assert load_schema()["title"] == "sparkbench.result.v1"


def test_examples_parse_and_legacy_hashes_match() -> None:
    for name in ("q4", "q6"):
        config = load_config(ROOT / "examples" / f"llamacpp-gb10-{name}-mtp7.toml")
        imported = read_json(V1 / f"main_{name}_mtp7.json")
        assert config.config_sha256 == imported["config_sha256"]
        assert config.workload_sha256 == imported["workload_sha256"]
        assert config.server is not None and config.server.command[0].endswith("llama-server")
        assert config.backend.extra_body == {"cache_prompt": False}
    for name in ("sglang-q4", "vllm-q4"):
        config = load_config(ROOT / "examples" / f"{name}.toml")
        assert config.backend.tokenizer == "openai-tokenize"
        assert config.backend.extra_body == {}


def test_workload_hash_ignores_machine_specific_fields() -> None:
    base = load_config(ROOT / "examples" / "llamacpp-gb10-q4-mtp7.toml").raw
    other = copy.deepcopy(base)
    other["model"]["file"] = "/srv/models/q4.gguf"
    other["runtime"]["binary"] = "/opt/llama.cpp/llama-server"
    other["backend"]["base_url"] = "http://spark-2:8080"
    other["server"]["command"][0] = "/opt/llama.cpp/llama-server"
    other["host"] = {"name": "spark-2"}
    a, b = parse_config(base), parse_config(other)
    assert a.workload_sha256 == b.workload_sha256
    assert a.config_sha256 != b.config_sha256
    other["parameters"]["mtp_depth"] = 10
    assert parse_config(other).workload_sha256 != a.workload_sha256


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"model": {"name": "m"}}, "base_url"),
        ({"backend": {"base_url": "http://x"}}, "[model] name"),
        ({"backend": {"base_url": "http://x"}, "model": {"name": "m"}, "bogus": {}}, "unknown section"),
        ({"backend": {"base_url": "http://x", "tokenizer": "sentencepiece"}, "model": {"name": "m"}}, "tokenizer"),
        ({"backend": {"base_url": "http://x"}, "model": {"name": "m", "typo": 1}}, "unknown key(s) in [model]"),
        ({"backend": {"base_url": "http://x"}, "model": {"name": "m"}, "run": {"suites": ["mmlu"]}}, "unknown suite"),
        ({"backend": {"base_url": "http://x"}, "model": {"name": "m"}, "server": {"command": "llama-server"}}, "command"),
        ({"backend": {"base_url": "http://x"}, "model": {"name": "m"}, "request": {"temperature": "0"}}, "number"),
    ],
)
def test_config_errors(raw: dict, message: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse_config(raw)
    assert message in str(info.value)


def test_config_defaults() -> None:
    config = parse_config({"backend": {"base_url": "http://x/"}, "model": {"name": "m"}})
    assert config.backend.base_url == "http://x"
    assert config.temperature == 0 and isinstance(config.temperature, int)
    assert config.seed == 42
    assert len(config.suites) == 7
    assert config.server is None
    assert config.label == "run" and config.output == "results/runs/run.json"
