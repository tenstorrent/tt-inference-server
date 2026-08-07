# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Every spec-test template must accept ``ctx``.

``dispatch._instantiate`` calls ``cls(config, targets, description=..., ctx=ctx)``
and falls back to ``cls(config, targets)`` on TypeError. That fallback is a
compatibility shim for older templates, but it fails silently: a template whose
``__init__`` forgets ``ctx`` still constructs, then quietly loses
``ctx.service_port``, ``ctx.base_url`` and its report ``block_id`` -- so its
result can be misattributed rather than reported as broken.

ImageGenerationLoraLoadTest was the one template in this state. This test pins
the contract for all of them so the shim stays unused.
"""

from __future__ import annotations

import inspect

import pytest

from test_module import eval_tests, load_param_tests
from test_module._test_common import TestConfig


def _template_classes():
    seen = {}
    for module in (load_param_tests, eval_tests):
        for name in getattr(module, "__all__", dir(module)):
            if not name.endswith("Test"):
                continue
            obj = getattr(module, name, None)
            if inspect.isclass(obj):
                seen.setdefault(name, obj)
    return sorted(seen.items())


TEMPLATES = _template_classes()


def test_templates_discovered():
    """Guard the guard: an empty list would make the test below vacuous."""
    assert len(TEMPLATES) >= 15, f"only found {len(TEMPLATES)} templates"


@pytest.mark.parametrize("name,cls", TEMPLATES, ids=[n for n, _ in TEMPLATES])
def test_template_accepts_ctx(name, cls):
    config = TestConfig(
        {
            "timeout": 10,
            "retry_attempts": 0,
            "retry_delay": 1,
            "break_on_failure": False,
        }
    )
    try:
        cls(config, {}, description="d", ctx=None)
    except TypeError as exc:
        if "ctx" in str(exc):
            pytest.fail(
                f"{name}.__init__ does not accept ctx, so dispatch._instantiate "
                f"silently falls back and drops ctx: {exc}"
            )
        # Some templates validate targets in __init__; that is not our concern.
    except Exception:
        # Only the ctx signature is under test here.
        pass
