# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The metal re-exports of tt_model_runners.minimax_h3_policy against an older metal python tree.

metal 34260b25483 (the pinned OM Quad3 tree) has ``policy.py`` with the aspect/duration constants
but without ``MINIMAX_H3_NUM_INFERENCE_STEPS`` and the two helpers the server later moved there.
Those must resolve to the server's legacy copies; whatever metal does define must win.
"""

import sys
import types

import pytest


@pytest.fixture
def old_metal_policy(monkeypatch):
    """A fake ``models.tt_dit.pipelines.minimax_h3`` package shaped like 34260b25483."""
    policy = types.ModuleType("models.tt_dit.pipelines.minimax_h3.policy")
    policy.MINIMAX_H3_ASPECT_RATIOS = (
        (16, 9),
        (9, 16),
        (2, 1),
    )  # distinctive: not the legacy tuple
    policy.MINIMAX_H3_DEFAULT_ASPECT_RATIO = (16, 9)
    policy.MINIMAX_H3_DURATIONS_S = tuple(range(4, 16))
    policy.MINIMAX_H3_DEFAULT_DURATION_S = 5
    packing = types.ModuleType("models.tt_dit.pipelines.minimax_h3.packing")
    packing.MINIMAX_H3_FRAMES_PER_CHUNK = 17
    packing.MINIMAX_H3_LATENTS_PER_CHUNK = 5
    pkg = types.ModuleType("models.tt_dit.pipelines.minimax_h3")
    pkg.policy = policy
    pkg.packing = packing
    for name, module in (
        ("models", types.ModuleType("models")),
        ("models.tt_dit", types.ModuleType("models.tt_dit")),
        ("models.tt_dit.pipelines", types.ModuleType("models.tt_dit.pipelines")),
        ("models.tt_dit.pipelines.minimax_h3", pkg),
        ("models.tt_dit.pipelines.minimax_h3.policy", policy),
        ("models.tt_dit.pipelines.minimax_h3.packing", packing),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return policy


def test_metal_names_win_when_present(old_metal_policy):
    from tt_model_runners import minimax_h3_policy as p

    assert p.MINIMAX_H3_ASPECT_RATIOS == ((16, 9), (9, 16), (2, 1))
    assert p.MINIMAX_H3_DURATIONS_S == tuple(range(4, 16))


def test_missing_names_fall_back_to_the_legacy_server_copies(old_metal_policy):
    from tt_model_runners import minimax_h3_policy as p

    assert p.MINIMAX_H3_NUM_INFERENCE_STEPS == 50
    # the legacy parser still restricts to the METAL aspect set, so the two cannot disagree
    assert p.minimax_h3_parse_aspect_ratio("2:1") == (2, 1)
    assert p.minimax_h3_parse_aspect_ratio(" 16x9 ") == (16, 9)
    with pytest.raises(ValueError, match="not served"):
        p.minimax_h3_parse_aspect_ratio("4:3")
    with pytest.raises(ValueError, match="W:H"):
        p.minimax_h3_parse_aspect_ratio("wide")
    assert p.minimax_h3_frames_are_aligned(124) and p.minimax_h3_frames_are_aligned(5)
    assert not p.minimax_h3_frames_are_aligned(
        125
    ) and not p.minimax_h3_frames_are_aligned(4)


def test_dit_runners_import_path_resolves(old_metal_policy):
    """The name dit_runners imports at module load must come back as a plain int."""
    from tt_model_runners.minimax_h3_policy import MINIMAX_H3_NUM_INFERENCE_STEPS

    assert MINIMAX_H3_NUM_INFERENCE_STEPS == 50


def test_unknown_names_still_raise():
    from tt_model_runners import minimax_h3_policy as p

    with pytest.raises(AttributeError):
        p.NOT_A_POLICY_ITEM  # noqa: B018
