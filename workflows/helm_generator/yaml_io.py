# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def _yaml() -> YAML:
    y = YAML(typ="rt")
    y.indent(mapping=2, sequence=4, offset=2)
    y.preserve_quotes = True
    y.width = 4096
    return y


def load_values(path: Path) -> Any:
    with open(path, "r") as f:
        return _yaml().load(f)


def dump_values(doc: Any, path: Path) -> None:
    with open(path, "w") as f:
        _yaml().dump(doc, f)


def dumps_values(doc: Any) -> str:
    from io import StringIO

    buf = StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()
