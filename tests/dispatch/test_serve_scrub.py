# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""_scrub_stale_tt_paths must drop stale TT_* input paths but keep output dirs like
TT_METAL_CACHE (which are created lazily and may not exist yet)."""

from tt_inference_server.dispatch.serve import _scrub_stale_tt_paths


def test_keeps_cache_var_even_when_missing():
    env = {
        "TT_METAL_CACHE": "/nonexistent/fresh-cache",        # output dir, not yet created
        "TT_MESH_GRAPH_DESC_PATH": "/old/checkout/mesh.yaml",  # stale input path
        "TT_GOOD_INPUT": "/",                                  # existing path
        "PATH": "/nonexistent/but-not-TT",                     # non-TT, untouched
    }
    removed = _scrub_stale_tt_paths(env)

    assert "TT_METAL_CACHE" in env                       # exempt output dir preserved
    assert "TT_MESH_GRAPH_DESC_PATH" not in env          # stale input scrubbed
    assert "TT_GOOD_INPUT" in env                        # existing path kept
    assert "PATH" in env                                 # non-TT untouched
    assert any("TT_MESH_GRAPH_DESC_PATH" in s for s in removed)
    assert all("TT_METAL_CACHE" not in s for s in removed)


def test_no_scrub_when_all_paths_exist():
    env = {"TT_METAL_CACHE": "/", "TT_FOO": "/"}
    assert _scrub_stale_tt_paths(env) == []
    assert env == {"TT_METAL_CACHE": "/", "TT_FOO": "/"}
