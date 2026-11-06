# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""lmms-eval plugin package: extra model clients for the EVALS_VISION venv.

lmms-eval imports ``<package>.models`` for every package named in the
``LMMS_EVAL_PLUGINS`` environment variable and registers each entry of its
``AVAILABLE_MODELS`` dict as a simple model, so ``--model <name>`` resolves to
``<package>.models.<name>.<class>``. ``lmms_plugin_env`` builds that environment
for the eval subprocess; the package is ``tt_lmms_plugins`` in this directory.
"""

import os
from pathlib import Path

# The plugin itself is the top-level package ``tt_lmms_plugins`` under this
# directory, so only this directory goes on the venv's PYTHONPATH (the repo root
# would shadow installed packages with its ``utils`` / ``tests`` directories).
PLUGIN_PACKAGE = "tt_lmms_plugins"
PLUGIN_ROOT = Path(__file__).resolve().parent


def lmms_plugin_env(env: dict, plugin_root=PLUGIN_ROOT) -> dict:
    """Return ``env`` with the TT plugin registered for lmms-eval.

    The eval subprocess runs the venv's ``lmms-eval`` entry point, whose
    ``sys.path`` does not include this repository, so the plugin's parent
    directory is added to PYTHONPATH and the package name to LMMS_EVAL_PLUGINS.
    """
    env = dict(env)
    plugins = [p for p in env.get("LMMS_EVAL_PLUGINS", "").split(",") if p]
    if PLUGIN_PACKAGE not in plugins:
        plugins.append(PLUGIN_PACKAGE)
    env["LMMS_EVAL_PLUGINS"] = ",".join(plugins)
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    root = str(plugin_root)
    if root not in paths:
        paths.insert(0, root)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
