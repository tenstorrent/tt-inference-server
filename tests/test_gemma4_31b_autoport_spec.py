#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Guards on the Gemma 4 31B agentic-autoport device spec.

The autoport once shipped ``DISABLE_METAL_OP_TIMEOUT=1``, which silently turned
off tt-inference-server's hang detection (and with it the automatic tt-triage
capture) for every run of this model.  A CI pass obtained that way cannot
distinguish "no hang" from "hangs are no longer detected", so the disable was
replaced with a raised threshold.  These tests keep it that way.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATHS = [
    REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml",
    REPO_ROOT / "workflows" / "model_specs" / "prod" / "llm.yaml",
]

# tt-inference-server's default, set by
# run_vllm_api_server.set_metal_timeout_env_vars.
DEFAULT_OP_TIMEOUT_SECONDS = 5.0


def _autoport_env_blocks(spec_path):
    """Every env_vars block belonging to the Gemma 4 31B autoport."""
    document = yaml.safe_load(spec_path.read_text())
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if (
                    key == "env_vars"
                    and isinstance(value, dict)
                    and "GEMMA4_31B_AUTOPORT_DIR" in value
                ):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return found


@pytest.mark.parametrize("spec_path", SPEC_PATHS, ids=lambda p: p.parent.name)
def test_autoport_spec_is_present(spec_path):
    assert _autoport_env_blocks(spec_path), f"no Gemma 4 31B autoport env_vars in {spec_path}"


@pytest.mark.parametrize("spec_path", SPEC_PATHS, ids=lambda p: p.parent.name)
def test_autoport_never_disables_the_hang_watchdog(spec_path):
    for env_vars in _autoport_env_blocks(spec_path):
        assert "DISABLE_METAL_OP_TIMEOUT" not in env_vars, (
            "DISABLE_METAL_OP_TIMEOUT suppresses hang detection and tt-triage capture "
            "for this model; raise TT_METAL_OPERATION_TIMEOUT_SECONDS instead"
        )


@pytest.mark.parametrize("spec_path", SPEC_PATHS, ids=lambda p: p.parent.name)
def test_autoport_raises_the_op_timeout_above_the_default(spec_path):
    for env_vars in _autoport_env_blocks(spec_path):
        raw = env_vars.get("TT_METAL_OPERATION_TIMEOUT_SECONDS")
        assert raw is not None, "the autoport must set an explicit op timeout"
        # Spec env values are forwarded to os.environ; a non-string forces
        # set_runtime_env_vars down its str() warning path.
        assert isinstance(raw, str), f"op timeout must be a string, got {type(raw).__name__}"
        assert float(raw) > DEFAULT_OP_TIMEOUT_SECONDS, (
            f"op timeout {raw}s does not exceed the {DEFAULT_OP_TIMEOUT_SECONDS}s default, "
            "so it would not help the cold-compile prefill it exists to cover"
        )
