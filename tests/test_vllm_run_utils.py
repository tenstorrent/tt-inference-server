#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import multiprocessing

import pytest

from utils.vllm_run_utils import create_model_symlink


def test_create_model_symlink_replaces_an_existing_link(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    create_model_symlink(links, "model", old)
    assert create_model_symlink(links, "model", new).resolve() == new
    assert [p.name for p in links.iterdir()] == ["model"]


def test_create_model_symlink_leaves_a_correct_link_alone(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    link = create_model_symlink(tmp_path, "model", weights)
    inode = link.lstat().st_ino
    assert create_model_symlink(tmp_path, "model", weights) == link
    assert link.lstat().st_ino == inode


def test_create_model_symlink_refuses_a_non_symlink(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (tmp_path / "model").mkdir()
    with pytest.raises(AssertionError, match="non-symlink"):
        create_model_symlink(tmp_path, "model", weights)


def _link_many(links, weights, rounds):
    for _ in range(rounds):
        create_model_symlink(links, "model", weights)


def test_create_model_symlink_survives_concurrent_engines(tmp_path):
    """Submesh engines each run model_setup at startup, so several processes relink the same path at once."""
    weights = tmp_path / "weights"
    weights.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    create_model_symlink(links, "model", weights)
    procs = [
        multiprocessing.Process(target=_link_many, args=(links, weights, 200))
        for _ in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
    assert [proc.exitcode for proc in procs] == [0] * 4
    assert (links / "model").resolve() == weights
    assert [p.name for p in links.iterdir()] == ["model"]
