# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Producer path: populate the tt-metal JIT cache for a model, then publish it as a
tt-kernel bundle (closes #1 / #26).

    python -m tt_inference_server.dispatch.release \\
        --model ~/dispatch/models/allenai-OLMo-1B-hf \\
        --repo  mando2222/olmo-1b-blackhole \\
        --weights allenai/OLMo-1B-hf --public

Flow:
  1. **Warmup** — run the model for a few tokens in a *subprocess* with a fresh
     ``TT_METAL_CACHE`` so exactly one (unambiguous, attributable) build_key subtree is
     produced, and the device is released when the subprocess exits.
  2. **Publish** — shell out to ``tt-kernel push --cache-dir <fresh>`` to package and
     upload that build_key subtree, optionally with a runner (packaged wheel or
     reference spec) and a ``--weights`` model reference.

This is glue over tt-kernel's native artifact system — it adds no new bundle format.
Pre-compiled kernels published this way are a cache *hit* on a matching consumer
(``install_subtree`` lands them in the consumer's tt-metal cache dir), so first run skips
JIT recompilation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

_WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."


def _warmup(model: str, tokens: int, max_seq: int) -> int:
    """Run the model briefly to JIT-compile its kernels into TT_METAL_CACHE, then exit.
    Invoked as a subprocess (see main) so the device lock is released before the push."""
    from tt_inference_server.dispatch import load_model

    handle = load_model(model, max_seq=max_seq, unsafe=True, trace_region_size=0)
    out = handle.generate(_WARMUP_PROMPT, max_new_tokens=tokens, chat=False)
    print(f"[release] warmup ok: generated {len(out)} chars", flush=True)
    return 0


def _push_argv(args, cache_dir: str) -> "list[str]":
    # Invoke tt-kernel as a module via the current interpreter so it works whether or not
    # the venv's bin/ (with the `tt-kernel` console script) is on PATH.
    argv = [
        sys.executable,
        "-m",
        "tt_kernel.cli",
        "push",
        args.repo,
        "--cache-dir",
        cache_dir,
        "--private" if args.private else "--public",
    ]
    if args.build_key is not None:
        argv += ["--build-key", str(args.build_key)]
    if args.weights:
        argv += ["--weights", args.weights]
    if args.runner_spec:
        argv += ["--runner-spec", args.runner_spec]
    for whl in args.python_package or []:
        argv += ["--python-package", whl]
    if args.entry_point:
        argv += ["--entry-point", args.entry_point]
    if args.runner_source:
        argv += ["--runner-source", args.runner_source]
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="release",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", required=True, help="HF id or local weights dir to warm up."
    )
    ap.add_argument(
        "--repo",
        help="Target bundle repo as namespace/name (required unless --warmup-only).",
    )
    ap.add_argument(
        "--weights",
        help="HF model repo id this bundle targets (recorded in the manifest).",
    )
    ap.add_argument(
        "--runner-spec",
        help="Runner as module:Class (packaged with --python-package, else reference).",
    )
    ap.add_argument(
        "--python-package",
        action="append",
        help="Runner wheel/sdist to ship (repeatable).",
    )
    ap.add_argument(
        "--entry-point",
        help="Entry-point name the wheel registers under tt_inference_server.runners.",
    )
    ap.add_argument(
        "--runner-source", help="For a reference runner: pip name / git URL."
    )
    ap.add_argument(
        "--tokens", type=int, default=8, help="Warmup tokens to generate (default 8)."
    )
    ap.add_argument(
        "--max-seq", type=int, default=512, help="Warmup max_seq (default 512)."
    )
    ap.add_argument(
        "--cache-dir", help="Cache root to compile into (default: a fresh temp dir)."
    )
    ap.add_argument(
        "--build-key",
        type=int,
        help="Explicit build_key to publish (only needed if ambiguous).",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Publish the repo private (default public).",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        help="Publish public (the default; accepted for explicitness).",
    )
    ap.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Assume the cache-dir is already populated.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Warm up but print the push command instead of running it.",
    )
    ap.add_argument(
        "--warmup-only", action="store_true", help=argparse.SUPPRESS
    )  # internal subprocess mode
    args = ap.parse_args(argv)

    if args.warmup_only:
        return _warmup(args.model, args.tokens, args.max_seq)
    if not args.repo:
        ap.error("--repo is required (the target namespace/name to publish to).")

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="ttk-release-")
    print(f"[release] cache dir: {cache_dir}", flush=True)

    if not args.skip_warmup:
        env = {**os.environ, "TT_METAL_CACHE": cache_dir}
        print(
            f"[release] warming up {args.model} ({args.tokens} tokens) ...", flush=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tt_inference_server.dispatch.release",
                "--warmup-only",
                "--model",
                args.model,
                "--tokens",
                str(args.tokens),
                "--max-seq",
                str(args.max_seq),
            ],
            env=env,
            check=True,
        )

    push = _push_argv(args, cache_dir)
    if args.dry_run:
        from tt_kernel import cache

        out_root = cache.resolve_out_root(cache_dir)
        print(
            "[release] DRY RUN — populated build_keys:",
            cache.list_build_keys(out_root),
            flush=True,
        )
        print("[release] would run:\n  " + " ".join(push), flush=True)
        return 0

    print("[release] publishing: " + " ".join(push), flush=True)
    return subprocess.run(push, check=True).returncode


if __name__ == "__main__":
    sys.exit(main())
