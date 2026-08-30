# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Validate a pinned Quetzal serving environment against the TT vLLM plugin."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import re
import token
import tokenize
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

LEGACY_QWEN_SOURCE_REVISION = "8a3bebe4afdd58068d4190248c3f7b82cc27ae9f"
PROFILE = "qwen36.serve"
MODEL_ID = "Qwen/Qwen3.6-27B"
CONTRACT_PATH = Path("serving/qualified_environments.json")
REQUIRED_NUMPY_SPECIFIER = ">=1.24.4,<2"
_TOML_TABLE = re.compile(r"^\s*\[([^]]+)]\s*(?:#.*)?$")


class ContractError(ValueError):
    """The selected Quetzal source cannot run in the official TTIS runtime."""


def _project_dependencies(project: Path) -> list[str]:
    """Parse only ``[project].dependencies`` without a host TOML dependency.

    The pre-image builder supports Python 3.10 and invokes this validator before
    dependency installation. Keep that bootstrap dependency-free and accept only
    the TOML string-array subset needed for Python package requirements. All other
    syntax fails closed rather than weakening the environment check.
    """

    try:
        lines = project.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read plugin project: {project}") from exc

    tables = [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := _TOML_TABLE.fullmatch(line)) is not None
    ]
    project_tables = [index for index, name in tables if name == "project"]
    if len(project_tables) != 1:
        raise ContractError("plugin project must contain exactly one [project] table")

    start = project_tables[0] + 1
    end = next((index for index, _ in tables if index >= start), len(lines))
    section = lines[start:end]
    assignments = [
        index
        for index, line in enumerate(section)
        if re.match(r"^\s*dependencies\s*=", line)
    ]
    if len(assignments) != 1:
        raise ContractError(
            "plugin [project] must contain exactly one dependencies assignment"
        )

    assignment = assignments[0]
    expression = section[assignment].split("=", 1)[1] + "\n"
    expression += "\n".join(section[assignment + 1 :])
    expression_lines = expression.splitlines()
    ignored = {
        token.ENDMARKER,
        token.INDENT,
        token.DEDENT,
        token.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    dependencies: list[str] = []
    started = False
    expect_value = True
    try:
        tokens = tokenize.generate_tokens(io.StringIO(expression).readline)
        for parsed in tokens:
            if parsed.type in ignored:
                continue
            if not started:
                if parsed.type != token.OP or parsed.string != "[":
                    raise ContractError(
                        "plugin [project].dependencies must be an array"
                    )
                started = True
                continue
            if parsed.type == token.OP and parsed.string == "]":
                suffix = expression_lines[parsed.end[0] - 1][parsed.end[1] :].strip()
                if suffix and not suffix.startswith("#"):
                    raise ContractError(
                        "plugin dependency array has unsupported trailing syntax"
                    )
                return dependencies
            if expect_value:
                if parsed.type != token.STRING:
                    raise ContractError(
                        "plugin dependencies must be literal requirement strings"
                    )
                try:
                    dependency = ast.literal_eval(parsed.string)
                except (SyntaxError, ValueError) as exc:
                    raise ContractError(
                        "plugin dependency is not a valid literal string"
                    ) from exc
                if not isinstance(dependency, str):
                    raise ContractError(
                        "plugin dependencies must be literal requirement strings"
                    )
                dependencies.append(dependency)
                expect_value = False
                continue
            if parsed.type != token.OP or parsed.string != ",":
                raise ContractError("plugin dependency array is missing a comma")
            expect_value = True
    except (IndentationError, tokenize.TokenError) as exc:
        raise ContractError(
            "plugin [project].dependencies contains unsupported TOML syntax"
        ) from exc

    raise ContractError("plugin [project].dependencies array is not terminated")


def _requirements_from_project(project: Path) -> dict[str, Requirement]:
    return {
        canonicalize_name(requirement.name): requirement
        for raw in _project_dependencies(project)
        for requirement in [Requirement(raw)]
    }


def _requirements_from_installed_plugin() -> dict[str, Requirement]:
    raw_requirements = importlib.metadata.requires("tt-vllm-plugin") or []
    return {
        canonicalize_name(requirement.name): requirement
        for raw in raw_requirements
        for requirement in [Requirement(raw)]
        if requirement.marker is None or requirement.marker.evaluate()
    }


def _validate_plugin_numpy_contract(
    requirements: dict[str, Requirement],
) -> Requirement:
    numpy_requirement = requirements.get("numpy")
    if numpy_requirement is None:
        raise ContractError("tt-vllm-plugin must declare an explicit NumPy requirement")
    expected = Requirement(f"numpy{REQUIRED_NUMPY_SPECIFIER}").specifier
    if numpy_requirement.specifier != expected:
        raise ContractError(
            "tt-vllm-plugin NumPy contract changed: expected "
            f"{REQUIRED_NUMPY_SPECIFIER}, found {numpy_requirement.specifier}"
        )
    return numpy_requirement


def _exact_version(distribution: str, raw: object) -> Version:
    if not isinstance(raw, str) or not re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)*(?:[a-z0-9.+-]*)?", raw
    ):
        raise ContractError(f"{PROFILE} must pin {distribution!r} to one exact version")
    try:
        return Version(raw)
    except InvalidVersion as exc:
        raise ContractError(
            f"{PROFILE} contains invalid {distribution!r} version {raw!r}"
        ) from exc


def validate_contract(
    source: Path,
    source_revision: str,
    *,
    plugin_project: Path | None = None,
    check_installed: bool = False,
) -> dict[str, object]:
    """Return a content-addressed receipt or raise before Quetzal is installed."""

    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ContractError("Quetzal source revision must be lowercase 40-hex")
    if not source.is_dir():
        raise ContractError(f"Quetzal source is not a directory: {source}")

    requirements = (
        _requirements_from_project(plugin_project)
        if plugin_project is not None
        else _requirements_from_installed_plugin()
    )
    numpy_requirement = _validate_plugin_numpy_contract(requirements)

    contract_path = source / CONTRACT_PATH
    if not contract_path.is_file():
        if source_revision != LEGACY_QWEN_SOURCE_REVISION:
            raise ContractError(
                "unpinned Quetzal source has no qualified serving environment; "
                f"only legacy source {LEGACY_QWEN_SOURCE_REVISION} is exempt"
            )
        if check_installed:
            installed_numpy = Version(importlib.metadata.version("numpy"))
            if installed_numpy not in numpy_requirement.specifier:
                raise ContractError(
                    f"legacy Qwen runtime NumPy {installed_numpy} violates "
                    f"tt-vllm-plugin {numpy_requirement.specifier}"
                )
        return {
            "schema": "ttis.quetzal-serve-environment.v1",
            "status": "legacy-pinned",
            "source_revision": source_revision,
            "profile": None,
        }

    raw_contract = contract_path.read_bytes()
    try:
        contract = json.loads(raw_contract)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("qualified environment contract is not valid JSON") from exc
    if contract.get("schema") != "quetzal.qualified-environments.v2":
        raise ContractError("unsupported Quetzal qualified-environments schema")

    variant = contract.get("variants", {}).get(PROFILE)
    if not isinstance(variant, dict):
        raise ContractError(f"Quetzal source must define {PROFILE!r}")
    if variant.get("lane") != "serve" or MODEL_ID not in variant.get("model_ids", []):
        raise ContractError(f"{PROFILE} must be the serve lane for {MODEL_ID}")

    base_dependencies = contract.get("base", {}).get("dependencies", {})
    overrides = variant.get("overrides", {})
    if not isinstance(base_dependencies, dict) or not isinstance(overrides, dict):
        raise ContractError(f"{PROFILE} dependencies must be mappings")
    qualified = {**base_dependencies, **overrides}
    if "numpy" not in qualified:
        raise ContractError(f"{PROFILE} must pin NumPy")

    exact_versions = {
        canonicalize_name(name): _exact_version(name, version)
        for name, version in qualified.items()
    }
    qualified_numpy = exact_versions["numpy"]
    if qualified_numpy not in numpy_requirement.specifier:
        raise ContractError(
            f"{PROFILE} pins numpy=={qualified_numpy}, outside tt-vllm-plugin "
            f"constraint {numpy_requirement.specifier}"
        )
    for name, exact in exact_versions.items():
        if name == "numpy":
            continue
        plugin_requirement = requirements.get(name)
        if plugin_requirement is not None and exact not in plugin_requirement.specifier:
            raise ContractError(
                f"{PROFILE} pins {name}=={exact}, outside tt-vllm-plugin "
                f"constraint {plugin_requirement.specifier}"
            )

    if check_installed:
        for name, exact in exact_versions.items():
            try:
                installed = Version(importlib.metadata.version(name))
            except importlib.metadata.PackageNotFoundError as exc:
                raise ContractError(
                    f"{PROFILE} requires installed {name}=={exact}"
                ) from exc
            if installed != exact:
                raise ContractError(
                    f"{PROFILE} requires {name}=={exact}; runtime has {installed}"
                )

    return {
        "schema": "ttis.quetzal-serve-environment.v1",
        "status": "qualified",
        "source_revision": source_revision,
        "profile": PROFILE,
        "qualified_environments_sha256": hashlib.sha256(raw_contract).hexdigest(),
        "dependencies": {
            name: str(version) for name, version in exact_versions.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--plugin-project", type=Path)
    parser.add_argument("--check-installed", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        receipt = validate_contract(
            args.source,
            args.source_revision,
            plugin_project=args.plugin_project,
            check_installed=args.check_installed,
        )
    except ContractError as exc:
        parser.error(str(exc))
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
