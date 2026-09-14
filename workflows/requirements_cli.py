# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared ``--requirements-json`` CLI plumbing for both entry points.

``run.py`` (full bring-up + dispatch) and ``run_workflows.py`` (standalone
engine) both accept an LLM-serving requirements document. Loading it,
defaulting ``--model``/``--device`` from it, relaxing the catalog gates, and
overlaying it onto the engine seams must behave identically whichever entry
point the operator uses -- and ``run.py`` dispatches *into* ``run_workflows.py``,
so a divergence would silently change the run halfway through. The rules
therefore live here rather than being duplicated per CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

REQUIREMENTS_FLAG = "--requirements-json"

# Device assumed for a requirements-driven run when neither --device nor the
# document's deployment.hardware is given. SUPER_CLUSTER matches a remote,
# horizontally-scaled OpenAI-compatible endpoint (token budget = context ×
# concurrency rather than a single-device KV pool), which is the target shape
# for requirements-driven validation.
DEFAULT_REQUIREMENTS_DEVICE = "super_cluster"

REQUIREMENTS_HELP = (
    "Path to an LLM-serving requirements document or validation plan "
    "(schemaVersion 2.x). Drives the run from the document: the accuracy evals "
    "it lists (gated by their reference scores/tolerances), the benchmark sweep "
    "points and their scalar targets/SLOs, and the model + deployment metadata "
    "(so a model not in the catalog can still be run). --model/--device default "
    "from the document when omitted. Accepts 'llm-gauntlet;<path>' to name a "
    "document by its path inside the llm-gauntlet repo, which is then cloned "
    "into this repo root -- e.g. "
    "'llm-gauntlet;specs/tt-internal/qwen3-32b/<id>.json'. That clone tracks "
    "the remote's default branch unless TT_LLM_GAUNTLET_REF names a branch, tag "
    "or commit (set TT_LLM_GAUNTLET_REF in CI so a run is reproducible)."
)


def requirements_mode_in_argv(argv: Sequence[str] | None = None) -> bool:
    """True if ``--requirements-json`` is present in ``argv``.

    Entry points must pre-scan argv because argparse evaluates ``choices`` and
    ``required`` at *definition* time: whether the catalog gate applies to
    ``--model``/``--device`` has to be known before the arguments are added.
    """
    scan = sys.argv[1:] if argv is None else argv
    return any(
        arg == REQUIREMENTS_FLAG or arg.startswith(f"{REQUIREMENTS_FLAG}=")
        for arg in scan
    )


def add_requirements_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--requirements-json`` on ``parser``."""
    parser.add_argument(
        REQUIREMENTS_FLAG, type=str, default=None, help=REQUIREMENTS_HELP
    )


def apply_requirements(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Load the document onto ``args.requirements_doc``; default model/device.

    Device precedence is ``--device`` > ``deployment.hardware`` > a default.
    The device does not decide which endpoint is hit (that is ``--server-url``)
    -- it only keys benchmark configs, token-budget math, and report labeling --
    so a requirements run against an already-running server can omit hardware
    entirely.
    """
    from workflow_module.requirements_schema import RequirementsError, load_requirements
    from workflows.llm_gauntlet_repo import (
        LLMGauntletError,
        resolve_requirements_location,
    )
    from workflows.model_spec_provider import hardware_to_device_name
    from workflows.requirements_target_pack import unknown_eval_names

    try:
        # A "llm-gauntlet;<path>" value names the document by its path inside
        # that repo; resolve it to a real file first. Done here, before the
        # load, so the absolutized path below is what every downstream consumer
        # sees -- RuntimeConfig, the forwarded child argv, the launcher re-exec.
        # None of them need to know the scheme exists.
        args.requirements_json = resolve_requirements_location(args.requirements_json)
        doc = load_requirements(args.requirements_json)
    except (LLMGauntletError, RequirementsError) as e:
        parser.error(str(e))
    # Reject unknown accuracy evals now rather than at eval-config build time,
    # halfway through the run.
    unknown = unknown_eval_names(doc)
    if unknown:
        parser.error(
            f"requirements document {args.requirements_json} names unknown "
            f"accuracy eval(s): {sorted(unknown)}. Add a mapping in "
            "workflows/requirements_target_pack.py:_EVAL_NAME_TO_TASK."
        )
    args.requirements_doc = doc
    # The path is forwarded to child processes and recorded in the runtime
    # config, so pin it to an absolute path rather than one relative to
    # whichever directory the operator happened to launch from.
    args.requirements_json = str(Path(args.requirements_json).expanduser().resolve())

    if not getattr(args, "device", None):
        hardware = doc.deployment.hardware
        if hardware:
            try:
                args.device = hardware_to_device_name(hardware).lower()
            except ValueError as e:
                parser.error(str(e))
        else:
            args.device = DEFAULT_REQUIREMENTS_DEVICE
            logger.warning(
                "No --device and no deployment.hardware in the requirements "
                "document; defaulting device to %r. Pass --device to override.",
                args.device,
            )
    if not getattr(args, "model", None):
        args.model = doc.model.name


def register_requirements_providers(doc: Any) -> None:
    """Overlay the document onto the engine seams for this process.

    Both wrappers delegate to the stock Tenstorrent implementations, so
    anything the document does not specify still falls through to the catalog.
    Every process that touches model resolution or validation content needs
    this -- ``run.py``, the launchers, and ``run_workflows.py`` each register
    independently, since they are separate processes.
    """
    from workflow_module.model_catalog import register_model_spec_provider
    from workflow_module.target_pack import register_target_pack
    from workflows.model_spec_provider import TenstorrentModelSpecProvider
    from workflows.requirements_target_pack import (
        RequirementsModelSpecProvider,
        RequirementsTargetPack,
    )
    from workflows.target_pack_provider import TenstorrentTargetPack

    register_model_spec_provider(
        RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    )
    register_target_pack(RequirementsTargetPack(doc, TenstorrentTargetPack()))


__all__ = [
    "DEFAULT_REQUIREMENTS_DEVICE",
    "REQUIREMENTS_FLAG",
    "REQUIREMENTS_HELP",
    "add_requirements_argument",
    "apply_requirements",
    "register_requirements_providers",
    "requirements_mode_in_argv",
]
