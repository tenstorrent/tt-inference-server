# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared immutable checkpoint and tokenizer identity handling."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _field(value, name, default=None):
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def get_pinned_revision(spec, *, required=False):
    """Return a matching immutable checkpoint/tokenizer pin, or None if unpinned."""
    device = _field(spec, "device_model_spec", {})
    args = _field(device, "vllm_args", {}) or {}
    metadata = _field(spec, "metadata", {}) or {}
    revision = args.get("revision")
    tokenizer = args.get("tokenizer_revision")
    if (
        revision is None
        and tokenizer is None
        and not (required or metadata.get("pinned_checkpoint", False))
    ):
        return None
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or tokenizer != revision
    ):
        raise ValueError(
            "Pinned artifacts require matching full checkpoint/tokenizer revision pins (immutable lowercase 40-hex commits)"
        )
    return revision


def resolve_tokenizer(
    model_spec, output_dir: Path, *, require_hashes: bool = False
) -> str:
    """Resolve the pinned tokenizer and save its verified identity with results."""
    from huggingface_hub import snapshot_download

    try:
        revision = get_pinned_revision(model_spec, required=True)
    except ValueError as exc:
        raise ValueError(
            "A pinned tokenizer requires matching checkpoint/tokenizer revision pins"
        ) from exc
    hashes = model_spec.metadata.get("benchmark_tokenizer_sha256", {})
    names = {"tokenizer.json", "tokenizer_config.json"}
    if (require_hashes or hashes) and set(hashes) != names:
        raise ValueError(
            "Fixed-workload references require a pinned tokenizer and hashes"
        )
    path = Path(
        snapshot_download(
            repo_id=model_spec.hf_model_repo,
            revision=revision,
            allow_patterns=sorted(names),
        )
    )
    actual = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names
    }
    if hashes and actual != hashes:
        raise ValueError("Tokenizer files differ from the frozen reference")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer_identity.json").write_text(
        json.dumps(
            {
                "repo": model_spec.hf_model_repo,
                "revision": revision,
                "path": str(path),
                "sha256": actual,
            },
            indent=2,
        )
        + "\n"
    )
    return str(path)
