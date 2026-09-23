# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Resolve ``--requirements-json llm-gauntlet;<path>`` against llm-gauntlet.

Requirements documents live in the llm-gauntlet repo, so a run names one by its
path inside that repo:

    --requirements-json "llm-gauntlet;specs/tt-internal/qwen3-32b/<id>.json"

Only ``specs/**/*.json`` is fetched, from the GitHub tarball API (no clone),
into ``<repo root>/llm-gauntlet``. ``TT_LLM_GAUNTLET_REF`` picks the branch,
tag or SHA (default ``main``); ``TT_LLM_GAUNTLET_TOKEN`` authenticates, which
the private repo requires. A value without the prefix is returned untouched.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

# ";" rather than ":" so path helpers don't read "x:y" as a drive/scheme.
LLM_GAUNTLET_PREFIX = "llm-gauntlet;"

# Directory the specs land in, under the repo root.
LLM_GAUNTLET_DIRNAME = "llm-gauntlet"

LLM_GAUNTLET_REPO = "tenstorrent/llm-gauntlet"
DEFAULT_LLM_GAUNTLET_REF = "main"
SPECS_DIRNAME = "specs"

LLM_GAUNTLET_REF_ENV = "TT_LLM_GAUNTLET_REF"
LLM_GAUNTLET_TOKEN_ENV = "TT_LLM_GAUNTLET_TOKEN"

DOWNLOAD_TIMEOUT_S = 120


class LLMGauntletError(RuntimeError):
    """Raised when fetching llm-gauntlet specs or resolving a path fails."""


def is_llm_gauntlet_ref(value: Optional[str]) -> bool:
    """True if ``value`` names a document by llm-gauntlet repo path."""
    return bool(value) and str(value).startswith(LLM_GAUNTLET_PREFIX)


def download_dir() -> Path:
    """``<repo root>/llm-gauntlet``; the VERSION marker also works without .git."""
    from workflows.utils import get_repo_root_path

    return get_repo_root_path(marker="VERSION") / LLM_GAUNTLET_DIRNAME


def resolve_ref(cli_ref: Optional[str] = None) -> str:
    """Ref to fetch: the argument, else the env var, else ``main``."""
    return cli_ref or os.getenv(LLM_GAUNTLET_REF_ENV) or DEFAULT_LLM_GAUNTLET_REF


def tarball_url(ref: str) -> str:
    return (
        f"https://api.github.com/repos/{LLM_GAUNTLET_REPO}/tarball/"
        f"{urllib.parse.quote(ref, safe='/')}"
    )


def _spec_member_path(member: tarfile.TarInfo) -> Optional[PurePosixPath]:
    """Path of ``member`` with the top-level dir stripped, if it is a spec json."""
    parts = PurePosixPath(member.name).parts[1:]
    if (
        not member.isfile()
        or len(parts) < 2
        or parts[0] != SPECS_DIRNAME
        or not parts[-1].endswith(".json")
        or ".." in parts
    ):
        return None
    return PurePosixPath(*parts)


def fetch_specs(dest: Path, ref: str, token: Optional[str] = None) -> int:
    """Replace ``dest`` with llm-gauntlet's ``specs/**/*.json`` at ``ref``.

    Returns the number of files written. Always re-downloads, since a branch
    moves. Staged in a temp dir so a failed download keeps the previous copy.
    """
    request = urllib.request.Request(
        tarball_url(ref),
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if token:
        # Unredirected: the API 302s to a pre-signed codeload URL.
        request.add_unredirected_header("Authorization", f"Bearer {token}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / LLM_GAUNTLET_DIRNAME
        staging.mkdir()
        count = 0
        try:
            with urllib.request.urlopen(
                request, timeout=DOWNLOAD_TIMEOUT_S
            ) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
                for member in tar:
                    rel = _spec_member_path(member)
                    if rel is None:
                        continue
                    out = staging / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    count += 1
        except (urllib.error.URLError, tarfile.TarError, OSError) as e:
            raise LLMGauntletError(
                f"Could not download llm-gauntlet specs (ref={ref}): {e}. "
                f"llm-gauntlet is private: set {LLM_GAUNTLET_TOKEN_ENV} to a "
                f"GitHub token with contents:read on it, and {LLM_GAUNTLET_REF_ENV} "
                f"if the document is not on {DEFAULT_LLM_GAUNTLET_REF}."
            ) from e
        if not count:
            raise LLMGauntletError(
                f"llm-gauntlet at ref={ref} has no {SPECS_DIRNAME}/**/*.json files."
            )
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(staging), str(dest))
    return count


def resolve_requirements_location(value: str, *, ref: Optional[str] = None) -> str:
    """Resolve ``llm-gauntlet;<path>`` to a real path, downloading the specs.

    Any other value is returned unchanged, without touching the network.
    """
    if not is_llm_gauntlet_ref(value):
        return value

    rel = value[len(LLM_GAUNTLET_PREFIX) :].strip().lstrip("/")
    if not rel:
        raise LLMGauntletError(
            f"{value!r} names no document. Expected "
            f"'{LLM_GAUNTLET_PREFIX}<path inside the llm-gauntlet repo>', e.g. "
            f"'{LLM_GAUNTLET_PREFIX}specs/<customer>/<model>/<id>.json'."
        )
    if Path(rel).is_absolute():
        raise LLMGauntletError(
            f"{value!r} must give a path *inside* the llm-gauntlet repo, not an "
            "absolute one. Drop the scheme prefix to use an absolute path."
        )

    dest = download_dir()
    resolved_ref = resolve_ref(ref)
    token = os.getenv(LLM_GAUNTLET_TOKEN_ENV) or None
    # Runs during argument parsing; without this a slow download looks like a hang.
    logger.info(
        "Resolving %s from llm-gauntlet (ref=%s, auth=%s) into %s ...",
        rel,
        resolved_ref,
        "token" if token else "none",
        dest,
    )
    fetch_specs(dest, resolved_ref, token=token)

    target = (dest / rel).resolve()
    root = dest.resolve()
    if root != target and root not in target.parents:
        raise LLMGauntletError(
            f"{value!r} resolves to {target}, outside the llm-gauntlet download "
            f"at {root}. Give a path inside the repo."
        )
    if not target.exists():
        raise LLMGauntletError(
            f"{value!r} not found in llm-gauntlet at ref={resolved_ref}. Only "
            f"{SPECS_DIRNAME}/**/*.json is fetched."
        )
    if target.is_dir():
        target = _resolve_document_in_dir(value, target)
    return str(target)


def _resolve_document_in_dir(value: str, directory: Path) -> Path:
    """Pick the document when ``value`` names a directory, not a file.

    Looks only at ``directory`` itself (no recursion): one json -> use it;
    several -> the alphabetically first, with a warning, since silently
    picking among ambiguous candidates should not pass without a trace.
    """
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        raise LLMGauntletError(
            f"{value!r} resolves to directory {directory} with no .json file "
            f"in it. Name the document directly, e.g. '{value.rstrip('/')}/<id>.json'."
        )
    if len(json_files) > 1:
        logger.warning(
            "%r resolves to directory %s containing %d json files (%s); "
            "picking %s. Name the document directly to avoid relying on this.",
            value,
            directory,
            len(json_files),
            ", ".join(f.name for f in json_files),
            json_files[0].name,
        )
    return json_files[0]


__all__ = [
    "DEFAULT_LLM_GAUNTLET_REF",
    "LLM_GAUNTLET_DIRNAME",
    "LLM_GAUNTLET_PREFIX",
    "LLM_GAUNTLET_REF_ENV",
    "LLM_GAUNTLET_REPO",
    "LLM_GAUNTLET_TOKEN_ENV",
    "LLMGauntletError",
    "download_dir",
    "fetch_specs",
    "is_llm_gauntlet_ref",
    "resolve_ref",
    "resolve_requirements_location",
    "tarball_url",
]
