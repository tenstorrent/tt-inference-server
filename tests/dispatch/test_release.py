# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the WS5 producer's push-command assembly (no hardware, no network)."""
import argparse
import sys

from tt_inference_server.dispatch import release


def _args(**kw):
    base = dict(repo="ns/m", build_key=None, weights=None, runner_spec=None,
                python_package=None, entry_point=None, runner_source=None,
                private=False, public=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_push_argv_kernels_only_public():
    argv = release._push_argv(_args(), "/cache")
    assert argv == [sys.executable, "-m", "tt_kernel.cli", "push", "ns/m",
                    "--cache-dir", "/cache", "--public"]


def test_push_argv_private_with_weights_and_build_key():
    argv = release._push_argv(_args(private=True, weights="org/w", build_key=42), "/c")
    assert "--private" in argv and "--public" not in argv
    assert argv[argv.index("--weights") + 1] == "org/w"
    assert argv[argv.index("--build-key") + 1] == "42"


def test_push_argv_packaged_runner():
    argv = release._push_argv(
        _args(runner_spec="pkg.mod:R", python_package=["/a.whl", "/b.whl"], entry_point="r"),
        "/c",
    )
    assert argv[argv.index("--runner-spec") + 1] == "pkg.mod:R"
    assert argv.count("--python-package") == 2
    assert argv[argv.index("--entry-point") + 1] == "r"


def test_push_argv_reference_runner_source():
    argv = release._push_argv(_args(runner_spec="pkg:R", runner_source="git+https://x"), "/c")
    assert argv[argv.index("--runner-source") + 1] == "git+https://x"
    assert "--python-package" not in argv  # reference runner ships no wheel
