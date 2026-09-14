# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Resolve ``--requirements-json llm-gauntlet;<path>`` against a clone.

Requirements documents and the validation plans exported from them live in the
llm-gauntlet repo, not here. Copying one into a runner's working directory to
start a run loses the only thing that makes a result citable -- which document,
at which revision, produced it. This scheme lets an operator or a CI job name
the document by its path *inside that repo*:

    --requirements-json "llm-gauntlet;specs/tt-internal/qwen3-32b/<id>.json"

The repo is cloned into the tt-inference-server root and the path resolved
inside it. Unlike the pinned checkouts in ``workflows/workflow_venvs.py`` this
tracks the remote's default branch: a requirements document is the customer's
statement of intent, and a run should validate against what that statement says
now, not against whatever it said when a pin was last bumped. Use
``TT_LLM_GAUNTLET_REF`` to pin a branch or SHA when reproducibility matters
more, which is what CI should do. The clone is over SSH by default; set
``TT_LLM_GAUNTLET_TOKEN`` on a CI runner, which has a token rather than a key,
and it switches to authenticated HTTPS.

A value without the prefix is returned untouched, so an ordinary path costs
nothing and reaches no git code at all.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Scheme prefix. ";" rather than ":" because a Windows-style or URL-ish "x:y"
# reads as a drive/scheme and invites path helpers to mangle it.
LLM_GAUNTLET_PREFIX = "llm-gauntlet;"

# Directory name the clone lands in, under the repo root.
LLM_GAUNTLET_DIRNAME = "llm-gauntlet"

# llm-gauntlet is private, so anonymous access fails either way and the
# transport is decided by which credential the host actually has. SSH by
# default: a developer machine and any runner that checks this repo out over
# SSH already has a usable key. A GitHub Actions runner usually has neither a
# key nor a credential helper -- it authenticates with a token -- so setting
# LLM_GAUNTLET_TOKEN_ENV switches to HTTPS, where a token applies.
DEFAULT_LLM_GAUNTLET_REPO = "git@github.com:tenstorrent/llm-gauntlet.git"
DEFAULT_LLM_GAUNTLET_REPO_HTTPS = "https://github.com/tenstorrent/llm-gauntlet.git"

# Env overrides. The ref one matters because launchers re-exec with argv
# verbatim and CI wants one setting to cover every child process; the repo one
# forces a transport or points at a mirror.
LLM_GAUNTLET_REPO_ENV = "TT_LLM_GAUNTLET_REPO"
LLM_GAUNTLET_REF_ENV = "TT_LLM_GAUNTLET_REF"

# A GitHub token with read access to llm-gauntlet contents. Setting it selects
# HTTPS and authenticates the fetch. Deliberately a separate variable rather
# than asking operators to embed the token in the URL: the token is sent as a
# git *config* value through the environment, so it never appears in argv (and
# therefore never in run_command's command log), and the URL stays printable.
LLM_GAUNTLET_TOKEN_ENV = "TT_LLM_GAUNTLET_TOKEN"


class LLMGauntletError(RuntimeError):
    """Raised when the llm-gauntlet clone or path resolution fails."""


def is_llm_gauntlet_ref(value: Optional[str]) -> bool:
    """True if ``value`` names a document by llm-gauntlet repo path."""
    return bool(value) and str(value).startswith(LLM_GAUNTLET_PREFIX)


def clone_dir() -> Path:
    """Where the clone lives: ``<repo root>/llm-gauntlet``.

    Uses the ``VERSION`` marker rather than ``.git``: a release tarball or an
    image layer has no ``.git``, and ``get_repo_root_path`` raises there. Same
    reason ``get_version`` uses it.
    """
    from workflows.utils import get_repo_root_path

    return get_repo_root_path(marker="VERSION") / LLM_GAUNTLET_DIRNAME


def resolve_ref(cli_ref: Optional[str] = None) -> Optional[str]:
    """Ref to check out: the flag, else the env var, else the default branch."""
    return cli_ref or os.getenv(LLM_GAUNTLET_REF_ENV) or None


def redact_url(url: str) -> str:
    """``url`` with any embedded credential replaced, safe to log.

    Covers an operator who put a token straight in
    ``TT_LLM_GAUNTLET_REPO``; the token env var never reaches a URL at all.
    """
    scheme, sep, rest = url.partition("://")
    if not sep or "@" not in rest:
        return url
    return f"{scheme}{sep}***@{rest.split('@', 1)[1]}"


def _token_env(repo: str, token: Optional[str]) -> Optional[dict]:
    """Git environment carrying ``token`` as an auth header, or None.

    Uses ``GIT_CONFIG_*`` rather than an argv ``-c`` or a credential in the
    URL, so the token stays out of the command line that ``run_command``
    logs. This is the same ``http.<host>.extraheader`` mechanism
    ``actions/checkout`` uses.
    """
    if not token or not repo.startswith("https://"):
        return None
    host = repo.split("://", 1)[1].split("/", 1)[0]
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"http.https://{host}/.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
            # Never let git stop for a username/password prompt: on a runner
            # that hangs the job until the step timeout instead of failing.
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def checkout_repo_ref(
    dest: Path, repo: str, ref: Optional[str], token: Optional[str] = None
) -> bool:
    """Materialize ``repo`` at ``ref`` (or its default branch) in ``dest``.

    Deliberately *not* ``workflow_venvs.checkout_pinned_repo``: that helper is
    built around a concrete ref (``fetch --depth 1 origin <ref>`` then
    ``checkout FETCH_HEAD``), and no ref string means "whatever the remote's
    HEAD points at". Bending it would make a pinned-checkout helper lie about
    what it guarantees.

    An existing clone is always re-fetched, because a default branch moves --
    the opposite of ``checkout_pinned_repo``, where re-running with the same
    pin is meant to be a no-op. One shallow fetch is cheap.
    """
    from workflow_module.proc import run_command

    env = _token_env(repo, token)

    def git(command: str) -> int:
        return run_command(command, logger=logger, env=env)

    if not (dest / ".git").is_dir():
        if dest.exists():
            logger.info("Discarding non-git directory at %s", dest)
            shutil.rmtree(dest)
        branch = f" --branch {ref}" if ref else ""
        # Without --branch, --depth 1 clones the remote's default branch, so
        # there is no origin/HEAD lookup to get wrong.
        if git(f"git clone --depth 1{branch} {repo} {dest}"):
            return False
        if not ref:
            return True

    steps = (
        f"git -C {dest} remote set-url origin {repo}",
        # "origin HEAD" resolves the remote's default branch server-side, so
        # this keeps working across a default-branch rename and needs no local
        # origin/HEAD. A concrete ref (branch, tag or SHA) fetches directly.
        f"git -C {dest} fetch --depth 1 origin {ref or 'HEAD'}",
        f"git -C {dest} checkout --detach --force FETCH_HEAD",
    )
    for step in steps:
        if git(step):
            logger.error("Failed to check out %s in %s (%s)", ref or "HEAD", dest, step)
            return False
    return True


def resolve_requirements_location(
    value: str, *, ref: Optional[str] = None, repo: Optional[str] = None
) -> str:
    """Resolve ``llm-gauntlet;<path>`` to a real path, cloning if needed.

    Any other value is returned unchanged -- checked first, so a plain path
    never touches the network or even imports the git plumbing.
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

    dest = clone_dir()
    resolved_ref = resolve_ref(ref)
    token = os.getenv(LLM_GAUNTLET_TOKEN_ENV) or None
    # A token only authenticates HTTPS, so having one selects that transport --
    # setting the token alone is enough, with no second variable to remember.
    default_repo = (
        DEFAULT_LLM_GAUNTLET_REPO_HTTPS if token else DEFAULT_LLM_GAUNTLET_REPO
    )
    repo_url = repo or os.getenv(LLM_GAUNTLET_REPO_ENV) or default_repo
    # Logged before the clone: this runs during argument parsing, so without it
    # a slow fetch looks like the CLI hanging before it has printed anything.
    logger.info(
        "Resolving %s from llm-gauntlet (%s, ref=%s, auth=%s) into %s ...",
        rel,
        redact_url(repo_url),
        resolved_ref or "default branch",
        "token" if token else "ssh/ambient",
        dest,
    )
    if not checkout_repo_ref(dest, repo_url, resolved_ref, token=token):
        raise LLMGauntletError(
            f"Could not check out {redact_url(repo_url)} "
            f"(ref={resolved_ref or 'default branch'}) into {dest}. "
            f"llm-gauntlet is private, so git needs credentials that can read "
            f"it. On a CI runner set {LLM_GAUNTLET_TOKEN_ENV} to a GitHub token "
            f"with contents:read on llm-gauntlet (this switches to HTTPS); "
            f"elsewhere use an SSH key, or set {LLM_GAUNTLET_REPO_ENV} to "
            f"another URL. Set {LLM_GAUNTLET_REF_ENV} if the document is not on "
            f"the default branch."
        )

    target = (dest / rel).resolve()
    root = dest.resolve()
    # The loader imposes no base directory, because operators legitimately keep
    # documents anywhere. Here there *is* one, so a "../.." escape is a mistake
    # worth failing on rather than silently reading some unrelated file.
    if root != target and root not in target.parents:
        raise LLMGauntletError(
            f"{value!r} resolves to {target}, outside the llm-gauntlet clone at "
            f"{root}. Give a path inside the repo."
        )
    return str(target)


__all__ = [
    "DEFAULT_LLM_GAUNTLET_REPO",
    "DEFAULT_LLM_GAUNTLET_REPO_HTTPS",
    "LLM_GAUNTLET_DIRNAME",
    "LLM_GAUNTLET_PREFIX",
    "LLM_GAUNTLET_REF_ENV",
    "LLM_GAUNTLET_REPO_ENV",
    "LLM_GAUNTLET_TOKEN_ENV",
    "LLMGauntletError",
    "checkout_repo_ref",
    "clone_dir",
    "is_llm_gauntlet_ref",
    "redact_url",
    "resolve_ref",
    "resolve_requirements_location",
]
