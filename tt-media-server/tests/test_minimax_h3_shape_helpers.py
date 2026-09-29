# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The H3 runner takes `align_num_frames` from wherever the installed tt-metal defines it:
`minimax_h3.packing` up to tt-metal main, `minimax_h3.policy` on the robustness line."""

import sys
import types

import pytest

from tt_model_runners import minimax_h3_policy


def _align(n):
    return n


@pytest.fixture
def minimax_h3_modules(monkeypatch):
    package = types.ModuleType("models.tt_dit.pipelines.minimax_h3")
    packing = types.ModuleType("models.tt_dit.pipelines.minimax_h3.packing")
    packing.MINIMAX_H3_FPS = 24
    packing.resolve_canvas_size = lambda w, h: (768, 1344)
    policy = types.ModuleType("models.tt_dit.pipelines.minimax_h3.policy")
    package.packing, package.policy = packing, policy
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, packing.__name__, packing)
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    return packing, policy


def test_align_num_frames_from_packing(minimax_h3_modules):
    packing, policy = minimax_h3_modules
    packing.align_num_frames = _align
    fps, align, canvas = minimax_h3_policy.minimax_h3_shape_helpers()
    assert (fps, align, canvas) == (24, _align, packing.resolve_canvas_size)


def test_align_num_frames_moved_to_policy(minimax_h3_modules):
    packing, policy = minimax_h3_modules
    policy.align_num_frames = _align
    fps, align, canvas = minimax_h3_policy.minimax_h3_shape_helpers()
    assert (fps, align, canvas) == (24, _align, packing.resolve_canvas_size)


def test_missing_everywhere_raises(minimax_h3_modules):
    with pytest.raises(ImportError):
        minimax_h3_policy.minimax_h3_shape_helpers()


def test_num_frames_from_metal_policy(minimax_h3_modules):
    packing, policy = minimax_h3_modules
    policy.get_num_frames = lambda seconds: 1000 + seconds
    assert minimax_h3_policy.minimax_h3_num_frames(5) == 1005


def test_num_frames_falls_back_without_get_num_frames(minimax_h3_modules):
    # tt-metal main: no policy.get_num_frames, align_num_frames still in packing.
    packing, policy = minimax_h3_modules
    packing.align_num_frames = lambda n: n + 1
    assert minimax_h3_policy.minimax_h3_num_frames(5) == 5 * 24 + 1
