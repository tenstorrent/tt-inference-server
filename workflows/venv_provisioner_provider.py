# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tenstorrent implementation of the engine's venv provisioning seam.

Adapts ``workflows/workflow_venvs.py``'s ``VENV_CONFIGS`` (uv-managed
per-tool environments) to
:class:`workflow_module.venv_provisioner.VenvProvisioner` so the engine
never imports the TT venv registry directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from workflow_module.venv_provisioner import VenvProvisioner
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)


class TenstorrentVenvProvisioner(VenvProvisioner):
    """``VenvProvisioner`` over the Tenstorrent ``VENV_CONFIGS`` registry."""

    def has_venv(self, venv_type: Any) -> bool:
        return venv_type in VENV_CONFIGS

    def venv_path(self, venv_type: Any) -> Path:
        return VENV_CONFIGS[venv_type].venv_path

    def venv_python(self, venv_type: Any) -> str:
        return str(VENV_CONFIGS[venv_type].venv_python)

    def provision(self, venv_type: Any, model_spec: Any = None) -> bool:
        # VenvConfig.setup runs uv venv -> extra_dirs -> requirements ->
        # setup_function hook, raising RuntimeError on failed steps.
        return bool(VENV_CONFIGS[venv_type].setup(model_spec=model_spec))
