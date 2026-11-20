# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""lmms-eval with the mimo_lmms plugin (llm_module/lmms_plugins/mimo_lmms) loaded.

Run with the EVALS_MULTIMODAL venv's python; every argument goes to lmms-eval's CLI.
The plugin's directory is put on sys.path and named in LMMS_EVAL_PLUGINS before
lmms-eval is imported (it registers plugin models at import time), so the venv
needs no install step for it. Its tasks (videomme_short) still need
``--include_path``: lmms-eval 0.7.4 adds a plugin's task directory only after it
has built the task index.
"""

import os
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent / "lmms_plugins"
PLUGIN_TASKS = PLUGIN_ROOT / "mimo_lmms" / "tasks"


def main() -> int:
    sys.path.insert(0, str(PLUGIN_ROOT))
    plugins = [p for p in os.environ.get("LMMS_EVAL_PLUGINS", "").split(",") if p]
    if "mimo_lmms" not in plugins:
        os.environ["LMMS_EVAL_PLUGINS"] = ",".join(plugins + ["mimo_lmms"])
    from lmms_eval.__main__ import cli_evaluate

    sys.argv = ["lmms-eval", *sys.argv[1:]]
    cli_evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
