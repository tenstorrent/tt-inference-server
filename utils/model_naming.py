# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Canonical translation between a model *identity* and a model *name token*.

A model has two distinct representations and they must not be confused:

===============  ===================  =====================================
representation   example              where it is used
===============  ===================  =====================================
data identity    ``Qwen/Qwen3-32B``   ``models-ci-config.json`` keys, report
                                      ``metadata``, HTTP ``model`` params,
                                      performance-target lookup, DB columns.
                                      Used verbatim -- never escaped.
name token       ``Qwen__Qwen3-32B``  filenames, directory names, GitHub
                                      artifact names, CI job names. Escaped.
===============  ===================  =====================================

Since the model id became the full HF repo id, every name built from a model
has to escape the org separator, and everything that reads such a name has to
undo the escape. Those two sides live in *different repositories* -- tt-shield
builds the artifact and job names, this repo parses them -- so the escape is a
cross-repo contract, and both directions belong in one place.

This module is that place. It is stdlib-only and imports nothing else from this
repo, so it is importable from any package here (including the deliberately
dependency-free ``report_module``) and runnable as a standalone script from a
plain checkout, with no ``pip install`` and no ``PYTHONPATH`` setup::

    python tt-inference-server/utils/model_naming.py slugify "Qwen/Qwen3-32B"

See ``docs/model_id_naming.md`` for the full contract and the tt-shield side.

Design notes
------------
The org prefix is **escaped, never stripped**, so a token stays unique and
round-trips: ``unslugify_model_id(slugify_model_id(x)) == x``.

``__`` rather than ``_`` is what makes that reversal exact. Model ids already
contain single underscores (``microsoft/phi-1_5``, ``yolox_nano``), so a
single-underscore separator is ambiguous and cannot be undone -- which is
exactly how the two sides drifted apart in the first place.

Known limitations, neither of which occurs in any current model id:

* a model id containing a literal ``__`` does not round-trip
  (``org/a__b`` -> ``org__a__b`` -> ``org/a/b``);
* whitespace maps to ``_`` and is not recovered.

A consumer that needs the identity back should prefer reading it from data --
report ``metadata.model_repo`` -- over reversing a name. Reverse a name only
when the name is all you have (an artifact listing, a directory scan).
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional, Tuple

__all__ = [
    "MODEL_ID_SEP",
    "ARTIFACT_FORBIDDEN_CHARS",
    "WORKFLOW_LOGS_PREFIX",
    "slugify_model_id",
    "unslugify_model_id",
    "slugify_name_parts",
    "model_name_variants",
    "is_artifact_name_safe",
    "workflow_logs_artifact_prefix",
    "split_workflow_logs_artifact_name",
    "ci_job_name",
    "device_from_ci_job_name",
    "ci_job_matches_device",
]

#: Escape sequence standing in for the HF org separator inside a name token.
MODEL_ID_SEP = "__"

#: Characters ``actions/upload-artifact`` rejects in an artifact name.
ARTIFACT_FORBIDDEN_CHARS = '/\\:<>|*?"\r\n'

#: Leading component of the per-run log bundle artifact name, produced by
#: tt-shield's ``workflow_run-tests-with-inference-server.yml``.
WORKFLOW_LOGS_PREFIX = "workflow_logs"


# ---------------------------------------------------------------------------
# core escape -- both directions
# ---------------------------------------------------------------------------
def slugify_model_id(model_id: str) -> str:
    """Escape a model identity for use in a filename, path or artifact name.

    Replaces the HF org separator with ``__`` and whitespace with ``_``,
    preserving the org prefix. Safe for POSIX paths and for
    ``actions/upload-artifact``.

    >>> slugify_model_id("Qwen/Qwen3-32B")
    'Qwen__Qwen3-32B'
    >>> slugify_model_id("resnet-50")
    'resnet-50'
    """
    if not model_id:
        return ""
    return (
        model_id.replace("/", MODEL_ID_SEP)
        .replace("\\", MODEL_ID_SEP)
        .replace(" ", "_")
    )


def unslugify_model_id(slug: str) -> str:
    """Recover a model identity from a token produced by :func:`slugify_model_id`.

    Ids that carry no org prefix are returned unchanged, so this is safe to
    apply to a token whose provenance is unknown.

    >>> unslugify_model_id("Qwen__Qwen3-32B")
    'Qwen/Qwen3-32B'
    >>> unslugify_model_id("microsoft__phi-1_5")
    'microsoft/phi-1_5'
    >>> unslugify_model_id("resnet-50")
    'resnet-50'
    """
    if not slug:
        return ""
    return slug.replace(MODEL_ID_SEP, "/")


def slugify_name_parts(*parts: Optional[str]) -> str:
    """Join non-empty ``parts`` with ``_`` and escape the result.

    The shape used for composite names -- report ids and report block ids
    (``<model>_<device>``). Empty and ``None`` parts are dropped, so a missing
    device does not leave a dangling separator.

    >>> slugify_name_parts("Qwen/Qwen3-32B", "p150")
    'Qwen__Qwen3-32B_p150'
    >>> slugify_name_parts("Qwen/Qwen3-32B", None)
    'Qwen__Qwen3-32B'
    """
    kept = [p for p in parts if p]
    if not kept:
        return ""
    return slugify_model_id("_".join(kept))


def model_name_variants(model_id: str) -> Tuple[str, ...]:
    """Every token a producer may plausibly have used for ``model_id``.

    Ordered most- to least-canonical and de-duplicated. Readers match against
    this instead of the canonical token alone, so a name survives producers
    that have not adopted this module yet:

    1. ``Qwen__Qwen3-32B`` -- canonical.
    2. ``Qwen/Qwen3-32B``  -- unescaped. Cannot occur in an artifact name
       (GitHub rejects ``/``) but does occur in job names, which allow it.
    3. ``Qwen_Qwen3-32B``  -- the single-underscore escape tt-shield's shell
       step used before this contract existed.
    4. ``Qwen3-32B``       -- the bare name model ids had before they became
       full HF repo ids. Last, because it is the only ambiguous one: two orgs
       can share a basename. No two models in ``models-ci-config.json``
       currently do.
    """
    if not model_id:
        return ()
    variants = [
        slugify_model_id(model_id),
        model_id,
        model_id.replace("/", "_").replace("\\", "_").replace(" ", "_"),
        model_id.rsplit("/", 1)[-1],
    ]
    seen = set()
    return tuple(v for v in variants if v and not (v in seen or seen.add(v)))


def is_artifact_name_safe(name: str) -> bool:
    """True if ``name`` is acceptable to ``actions/upload-artifact``.

    For asserting on the producing side, where the alternative is a job that
    runs to completion and then fails on upload.
    """
    return bool(name) and not any(c in name for c in ARTIFACT_FORBIDDEN_CHARS)


# ---------------------------------------------------------------------------
# CI name grammars -- shared with tt-shield
# ---------------------------------------------------------------------------
def workflow_logs_artifact_prefix(workflow: str, model_id: str) -> str:
    """``workflow_logs_<workflow>_<model>_`` -- the log bundle name prefix.

    The trailing ``_`` is part of the prefix so that models sharing a prefix
    (``foo`` vs ``foo-turbo``) stay unambiguous when filtering by
    ``str.startswith``.
    """
    return f"{WORKFLOW_LOGS_PREFIX}_{workflow}_{slugify_model_id(model_id)}_"


def split_workflow_logs_artifact_name(
    name: str, workflow: str, model_id: str
) -> Optional[Tuple[str, str]]:
    """Split a log bundle artifact name into ``(runner, suffix)``.

    Full grammar, as produced by tt-shield::

        workflow_logs_<workflow>_<model>_<runner_label>_<impl-or-default>

    Returns ``None`` if ``name`` is not this model's bundle. ``model_id`` and
    ``workflow`` are required because the grammar is not self-delimiting: the
    model token contains ``__`` and the runner label may contain ``_``, so the
    boundaries are only recoverable when the model is known. The suffix is
    taken from the last ``_``, leaving anything before it to the runner label.

    >>> split_workflow_logs_artifact_name(
    ...     "workflow_logs_release_Qwen__Qwen3-32B_p150_default",
    ...     "release", "Qwen/Qwen3-32B")
    ('p150', 'default')
    """
    if not name:
        return None
    for token in model_name_variants(model_id):
        prefix = f"{WORKFLOW_LOGS_PREFIX}_{workflow}_{token}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if "_" not in rest:
            return None
        runner, suffix = rest.rsplit("_", 1)
        return (runner, suffix) if runner else None
    return None


def ci_job_name(
    workflow: str, model_id: str, runner_label: str, runner_type: str
) -> str:
    """``run-<workflow>-<model>-<runner_label>-<runner_type>``.

    The per-matrix-entry job name. ``runner_type`` is the device, which is why
    this name is the only place the device of a (model, runner) pair can be
    recovered from -- it is absent from the artifact name.
    """
    return f"run-{workflow}-{slugify_model_id(model_id)}-{runner_label}-{runner_type}"


def device_from_ci_job_name(
    job_name: str, workflow: str, model_id: str, runner_label: str
) -> Optional[str]:
    """Recover the device from a job name built by :func:`ci_job_name`.

    ``None`` if ``job_name`` is not this (model, runner) pair's job.

    The GitHub jobs API prefixes a reusable workflow's job with its caller
    (``"caller job / run-release-..."``), so the marker is searched for rather
    than anchored at the start. It is deliberately *not* found by splitting on
    ``/`` first: an unescaped model id contains ``/``, and splitting would eat
    the org prefix along with the caller prefix.

    >>> device_from_ci_job_name(
    ...     "call-release / run-release-Qwen__Qwen3-32B-p150-P150",
    ...     "release", "Qwen/Qwen3-32B", "p150")
    'P150'
    """
    if not job_name:
        return None
    for token in model_name_variants(model_id):
        marker = f"run-{workflow}-{token}-{runner_label}-"
        idx = job_name.find(marker)
        if idx == -1:
            continue
        tail = job_name[idx + len(marker) :].strip()
        if not tail:
            continue
        # A device token has no spaces; trim the punctuation a wrapping job
        # name description can leave behind ("..., P150)").
        return tail.split()[0].strip(",)")
    return None


def _longest_matching_token(
    job_name: str, workflow: str, model_id: str, device_suffix: str
) -> Optional[int]:
    """Length of the longest model token that explains ``job_name``.

    ``None`` when no spelling of ``model_id`` matches. The length is what makes
    :func:`ci_job_matches_device` able to rank two models against one name.
    """
    best: Optional[int] = None
    for token in model_name_variants(model_id):
        marker = f"run-{workflow}-{token}-"
        idx = job_name.find(marker)
        if idx == -1:
            continue
        tail = job_name[idx + len(marker) :].strip().rstrip(",)")
        if tail.lower().endswith(device_suffix) and (best is None or len(token) > best):
            best = len(token)
    return best


def ci_job_matches_device(
    job_name: str,
    workflow: str,
    model_id: str,
    device: str,
    other_model_ids: Iterable[str] = (),
) -> bool:
    """True if ``job_name`` is the job for this ``(model, device)`` pair.

    :func:`device_from_ci_job_name` run backwards, for a caller that knows the
    device but not the runner label -- ``models-ci-config.json`` records the
    device and leaves the runner to tt-shield. Anchoring on both ends of the
    label leaves it unconstrained (``bh-qb-ge``); the device compares
    case-insensitively since callers hold a ``DeviceTypes`` name.

    Unlike the artifact grammar, this one separates fields with ``-``, which is
    also the commonest character *inside* a model name, so a prefix sibling's
    job is not distinguishable from this model's by string shape alone:
    ``Qwen/Qwen3-32B`` and ``Qwen/Qwen3-32B-FP8`` both explain
    ``run-release-Qwen__Qwen3-32B-FP8-p150-P150``. Pass the other models in
    scope as ``other_model_ids`` and the longer explanation wins, so the
    sibling's job is left to the sibling.

    >>> ci_job_matches_device(
    ...     "run-tests / run-release-meta-llama__Llama-3.3-70B-Instruct-bh-qb-ge-p300x2",
    ...     "release", "meta-llama/Llama-3.3-70B-Instruct", "P300X2")
    True
    >>> ci_job_matches_device(
    ...     "run-release-Qwen__Qwen3-32B-FP8-p150-P150",
    ...     "release", "Qwen/Qwen3-32B", "P150", ["Qwen/Qwen3-32B-FP8"])
    False
    """
    if not job_name or not device:
        return False
    # The leading "-" rejects a label-less name and "x2" vs "p300x2".
    suffix = f"-{device}".lower()
    own = _longest_matching_token(job_name, workflow, model_id, suffix)
    if own is None:
        return False
    for other in other_model_ids:
        if other == model_id:
            continue
        rival = _longest_matching_token(job_name, workflow, other, suffix)
        if rival is not None and rival > own:
            return False
    return True


# ---------------------------------------------------------------------------
# CLI -- for producers that build names in shell (tt-shield's YAML steps)
# ---------------------------------------------------------------------------
_USAGE = """usage: model_naming.py <command> [args]

  slugify <model_id>                            Qwen/Qwen3-32B -> Qwen__Qwen3-32B
  unslugify <slug>                              Qwen__Qwen3-32B -> Qwen/Qwen3-32B
  artifact-prefix <workflow> <model_id>         workflow_logs_<workflow>_<model>_
  job-name <workflow> <model_id> <label> <type> run-<workflow>-<model>-<label>-<type>

Writes the result to stdout, so a shell producer can do:

  SLUG=$(python tt-inference-server/utils/model_naming.py slugify "$MODEL")
"""

_COMMANDS = {
    "slugify": (1, lambda model: slugify_model_id(model)),
    "unslugify": (1, lambda slug: unslugify_model_id(slug)),
    "artifact-prefix": (2, workflow_logs_artifact_prefix),
    "job-name": (4, ci_job_name),
}


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(_USAGE)
        return 0
    command, rest = args[0], args[1:]
    entry = _COMMANDS.get(command)
    if entry is None:
        sys.stderr.write(f"error: unknown command {command!r}\n\n{_USAGE}")
        return 2
    argc, fn = entry
    if len(rest) != argc:
        sys.stderr.write(
            f"error: {command} takes {argc} argument(s), got {len(rest)}\n\n{_USAGE}"
        )
        return 2
    sys.stdout.write(f"{fn(*rest)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
