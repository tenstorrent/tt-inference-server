# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Write each generation deployment's generations API, as its OpenAPI document
serves it, to docs/api/openapi/<org>/<model>.yaml.

These files are the generations API's spec; tests/test_generations_api.py
checks that the server still serves exactly them. Run this after an intended
API change and review the diff:

    python scripts/dump_generations_openapi.py
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

SERVER_DIR = Path(__file__).resolve().parents[1]
SPEC_DIR = SERVER_DIR / "docs" / "api" / "openapi"

# The generations API's routes; the app's other routes (maintenance, models,
# metrics) and its title and version are not part of it.
_API_PATHS = ("/v1/generations/", "/v1/jobs")

# Read by config.settings besides its fields. A deployment's routes follow
# from MODEL_RUNNER alone.
_SETTINGS_ENV = (
    "MODEL",
    "MODEL_WEIGHTS_DIR",
    "USE_DYNAMIC_BATCHER",
    "SD_3_5_FAST",
    "SD_3_5_BASE",
    "TP2",
)

# Imports the app as main.py builds it for the runner and writes its document.
_DUMP = """
import json, sys
import main
with open(sys.argv[1], "w") as f:
    json.dump(main.app.openapi(), f)
"""


def _settings_fields() -> set[str]:
    """The fields of config.settings.Settings, read from its source: importing
    it would build the settings from this process's environment."""
    tree = ast.parse((SERVER_DIR / "config" / "settings.py").read_text())
    (settings,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    ]
    return {
        node.target.id
        for node in settings.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _environment(runner: str, metrics_dir: str) -> dict[str, str]:
    settings_names = {name.upper() for name in _settings_fields()}
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in settings_names and key not in _SETTINGS_ENV
    }
    # main.py clears its metrics directory on import: not a running server's.
    environment["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
    environment["MODEL_RUNNER"] = runner
    return environment


def _referenced_schemas(value) -> set[str]:
    return set(re.findall(r'"#/components/schemas/([^"]+)"', json.dumps(value)))


def generations_api(document: dict) -> dict:
    """The generations API's part of an app's OpenAPI document: its routes and
    the schemas they reference."""
    paths = {
        path: item
        for path, item in document["paths"].items()
        if path.startswith(_API_PATHS)
    }
    schemas = document["components"]["schemas"]
    needed, pending = set(), _referenced_schemas(paths)
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending |= _referenced_schemas(schemas[name])
    return {
        "openapi": document["openapi"],
        "paths": paths,
        "components": {
            **document["components"],
            "schemas": {name: schemas[name] for name in schemas if name in needed},
        },
    }


def served_generations_api(runner: str) -> dict:
    """The generations API the server serves with ``runner`` loaded.

    Built in a fresh process: the routers are assembled from the settings when
    open_ai_api is imported.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "openapi.json"
        process = subprocess.run(
            [sys.executable, "-c", _DUMP, str(path)],
            cwd=SERVER_DIR,
            env=_environment(runner, str(Path(directory) / "metrics")),
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"building the app for {runner} failed:\n{process.stderr}"
            )
        return generations_api(json.loads(path.read_text()))


def served_generations_api_by_runner(runners: list[str]) -> dict[str, dict]:
    if not runners:
        return {}
    with ThreadPoolExecutor(max_workers=len(runners)) as pool:
        return dict(zip(runners, pool.map(served_generations_api, runners)))


def spec_path(model_id: str) -> Path:
    return SPEC_DIR / f"{model_id}.yaml"


def spec_files() -> set[Path]:
    return set(SPEC_DIR.glob("*/*.yaml"))


def main() -> None:
    sys.path.insert(0, str(SERVER_DIR))
    from domain.generation_models import GENERATION_MODELS

    documents = served_generations_api_by_runner(list(GENERATION_MODELS))
    written = set()
    for runner, model in GENERATION_MODELS.items():
        path = spec_path(model.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                documents[runner], sort_keys=False, allow_unicode=True, width=88
            )
        )
        written.add(path)
        print(path.relative_to(SERVER_DIR))
    # A model no longer served keeps no spec.
    for path in spec_files() - written:
        path.unlink()
        print(f"removed {path.relative_to(SERVER_DIR)}")


if __name__ == "__main__":
    main()
