"""Minimal JSON Schema (draft 2020-12 subset) validator, stdlib only.

Supports exactly the keywords used by ``schemas/result.v1.json``; any other
keyword raises ``SchemaError`` so a schema edit can never be silently ignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "result.v1.json"

_ANNOTATIONS = {"$schema", "$id", "title", "description", "$defs", "default", "examples", "format"}
_KEYWORDS = {
    "$ref",
    "type",
    "const",
    "enum",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "pattern",
    "minimum",
    "maximum",
    "minLength",
    "anyOf",
    "oneOf",
    "allOf",
}

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
}


class SchemaError(ValueError):
    """The schema itself uses an unsupported construct."""


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name not in _TYPES:
        raise SchemaError(f"unsupported type {name!r}")
    return isinstance(value, _TYPES[name])


class Validator:
    def __init__(self, schema: dict[str, Any]) -> None:
        self.root = schema

    def _resolve(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#/"):
            raise SchemaError(f"only local refs are supported: {ref}")
        node: Any = self.root
        for part in ref[2:].split("/"):
            node = node[part]
        if not isinstance(node, dict):
            raise SchemaError(f"bad ref {ref}")
        return node

    def errors(self, value: Any) -> list[str]:
        found: list[str] = []
        self._check(value, self.root, "$", found)
        return found

    def _check(self, value: Any, schema: dict[str, Any] | bool, path: str, errors: list[str]) -> None:
        if schema is True or schema == {}:
            return
        if schema is False:
            errors.append(f"{path}: not allowed")
            return
        assert isinstance(schema, dict)
        unknown = set(schema) - _KEYWORDS - _ANNOTATIONS
        if unknown:
            raise SchemaError(f"unsupported keyword(s) {sorted(unknown)} at {path}")
        if "$ref" in schema:
            self._check(value, self._resolve(schema["$ref"]), path, errors)
        if "type" in schema:
            names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not any(_is_type(value, name) for name in names):
                errors.append(f"{path}: expected {'/'.join(names)}, got {type(value).__name__}")
                return
        if "const" in schema and value != schema["const"]:
            errors.append(f"{path}: expected constant {schema['const']!r}")
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: {value!r} not in {schema['enum']!r}")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path}: shorter than {schema['minLength']}")
            if "pattern" in schema and not re.search(schema["pattern"], value):
                errors.append(f"{path}: does not match {schema['pattern']}")
        if _is_type(value, "number"):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: {value} < minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: {value} > maximum {schema['maximum']}")
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"{path}: missing required property {key!r}")
            additional = schema.get("additionalProperties", True)
            for key, item in value.items():
                child = f"{path}.{key}"
                if key in properties:
                    self._check(item, properties[key], child, errors)
                elif additional is False:
                    errors.append(f"{path}: unexpected property {key!r}")
                elif isinstance(additional, dict):
                    self._check(item, additional, child, errors)
        if isinstance(value, list) and "items" in schema:
            for index, item in enumerate(value):
                self._check(item, schema["items"], f"{path}[{index}]", errors)
        if "allOf" in schema:
            for sub in schema["allOf"]:
                self._check(value, sub, path, errors)
        for keyword in ("anyOf", "oneOf"):
            if keyword not in schema:
                continue
            matches = 0
            for sub in schema[keyword]:
                sub_errors: list[str] = []
                self._check(value, sub, path, sub_errors)
                matches += not sub_errors
            if matches == 0 or (keyword == "oneOf" and matches > 1):
                errors.append(f"{path}: does not match {keyword} ({matches} alternatives matched)")


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError("schema root must be an object")
    return value


def validate(document: Any, schema: dict[str, Any] | None = None) -> list[str]:
    """Return a list of human-readable errors (empty when valid)."""
    return Validator(schema or load_schema()).errors(document)


def semantic_errors(document: Any) -> list[str]:
    """Consistency checks JSON Schema cannot express (run after schema checks)."""
    import hashlib

    errors: list[str] = []
    for name, suite in document.get("suites", {}).items():
        cases = suite["cases"]
        if suite["aggregate"]["cases"] != len(cases):
            errors.append(f"$.suites.{name}: aggregate.cases={suite['aggregate']['cases']} but {len(cases)} case records")
        counts = {key: sum(case["verdict"] == key for case in cases) for key in ("pass", "fail", "error", "unscored")}
        if counts != suite["verdicts"]:
            errors.append(f"$.suites.{name}: verdicts {suite['verdicts']} do not match case records {counts}")
        seen: set[str] = set()
        for index, case in enumerate(cases):
            if case["id"] in seen:
                errors.append(f"$.suites.{name}.cases[{index}]: duplicate id {case['id']!r}")
            seen.add(case["id"])
            content = case.get("content")
            if isinstance(content, str) and case.get("content_sha256") != hashlib.sha256(content.encode()).hexdigest():
                errors.append(f"$.suites.{name}.cases[{index}]: content_sha256 does not match content")
    return errors
