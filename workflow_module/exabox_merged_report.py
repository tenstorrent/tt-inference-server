# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Merge many per-test exabox reports into one release-shaped report.

Single-host CI runs ``run.py --workflow release``, so evals, benchmarks and
spec_tests all execute in one process and land in a single report. Exabox runs
each selected test as its own CI job against an already-deployed server, so a
run ends up with N single-kind reports and nothing that answers "how did this
model do overall".

This rebuilds that answer without re-running anything:

    discover_test_reports(container)  -> [Path]
      -> load_test_reports(...)       -> [SourceReport]
      -> build_merged_schema(...)     -> ReportSchema   (sections concatenated)
      -> acceptance_criteria_check    -> recomputed, model_status-gated
      -> ReportGenerator.generate     -> report_<id>.md + data/report_data_<id>.json

Inclusion is decided by SHAPE, never by test name, so a new exabox test needs no
change here:

* no ``sections`` -> skipped. This is what drops ``sanity``, whose results are
  a flat ad-hoc dict rather than a ReportSchema.

The evals smoke run IS merged, deliberately: collecting every result matters
more than the duplicate rows it produces. Smoke and full measure the same task
and match on ``kind``, ``title``, ``id`` and ``task_name``, so the merged
report shows two scores for one task with nothing to tell them apart.

Everything else is merged whether it passed or failed: a report that missed its
acceptance criteria is still a real measurement.

Run as::

    python3 -m workflow_module.exabox_merged_report \\
        --container-dir <dir of downloaded report artifacts> \\
        --output-dir <dir> [--model M] [--target T] [--job-id N] \\
        [--missing-test 'inference-workflow-benchmarks --dev-mode']
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module import (
    GenerateResult,
    ReportGenerator,
    ReportSchema,
    acceptance_criteria_check,
    build_acceptance_export,
    task_failure_blockers,
)

logger = logging.getLogger(__name__)

#: Only the job-id-suffixed copies written by tt-shield's "Prepare reports for
#: the data collector" step. run.py's own
#: ``workflow_logs/reports_output/<wf>/data/report_data_<model>_<ts>.json`` is
#: the same content under a different name, so matching both would duplicate
#: every section.
REPORT_GLOB = "report_*.json"

#: Also discovered, purely so the skip is visible in the coverage summary
#: rather than the file being silently absent. ``sanity`` writes a flat dict
#: with no ``sections``, so the shape filter drops it either way.
EXTRA_GLOBS = ("sanity_results_*.json",)

#: Kept for reference and for tests. No longer filters anything — see
#: mergeable() for why the evals smoke run is merged.
SMOKE_FLAG = "--limit-samples-mode smoke-test"

#: The merged report presents itself as a release, so every existing consumer
#: (dashboards, the release report renderer, anyone reading the file) treats it
#: exactly like a single-host one.
MERGED_WORKFLOW = "release"


@dataclass(frozen=True)
class SourceReport:
    """One per-test report, kept alongside the raw payload.

    ``ReportSchema`` carries only ``metadata`` and ``sections``, but the
    acceptance fields live at the *top level* of a written report — the
    generator hoists them out of metadata on the way to disk. Anything we need
    from them (model status, prior waivers) has to come from ``payload``.
    """

    path: Path
    schema: ReportSchema
    payload: Dict[str, Any]

    @property
    def job_id(self) -> int:
        """Trailing ``_<job_id>``; 0 when absent. Job ids rise over time, so
        this doubles as a recency key across CI attempts."""
        try:
            return int(self.path.stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0

    @property
    def workflow(self) -> str:
        return str(self.schema.metadata.get("workflow") or "")

    @property
    def run_command(self) -> str:
        return str(self.schema.metadata.get("run_command") or "")

    @property
    def model_status(self) -> str:
        meta = self.payload.get("acceptance_criteria_metadata") or {}
        return str(meta.get("model_status") or "")

    @property
    def generated_at(self) -> str:
        return str(self.schema.metadata.get("generated_at") or "")

    @property
    def identity(self) -> str:
        """What makes two reports "the same test", for de-duplication.

        The exact ``run.py`` invocation is the only field that separates tests
        reliably: the artifact name carries the test name but not its
        arguments, so evals smoke and evals full — and two ``custom`` runs with
        different commands — are indistinguishable by name alone.

        ``--server-url`` is stripped because a deployment recreated between CI
        attempts can hand out a different ngrok hostname for the same test,
        which would otherwise split one test into two identities and defeat the
        de-duplication.
        """
        command = _strip_server_url(self.run_command)
        if command:
            return command
        # No run_command: fall back to the workflow plus the tasks measured,
        # which is coarser but still separates unrelated tests.
        tasks = sorted(
            str((block.data or {}).get("task_name", ""))
            for block in self.schema.sections
            if isinstance(block.data, Mapping)
        )
        return f"{self.workflow}|{','.join(t for t in tasks if t)}"

    def waived(self) -> Dict[str, str]:
        """Blockers each source dropped because a known-issues waiver matched.

        Waivers come from the model spec, which is not available here, so
        ``acceptance_criteria_check`` below would re-raise them as real
        blockers. Carrying them forward keeps the merged verdict from being
        harsher than the individual tests it is built from.
        """
        out: Dict[str, str] = {}
        meta = self.payload.get("acceptance_criteria_metadata") or {}
        for category in meta.get("categories") or []:
            if isinstance(category, Mapping):
                waived = category.get("waived")
                if isinstance(waived, Mapping):
                    out.update({str(k): str(v) for k, v in waived.items()})
        return out


# --------------------------------------------------------------------------- #
# discovery and filtering
# --------------------------------------------------------------------------- #
def discover_test_reports(container: Path) -> List[Path]:
    """Every candidate report under ``container``, oldest job id first."""
    found: List[Path] = []
    for pattern in (REPORT_GLOB, *EXTRA_GLOBS):
        for path in sorted(container.rglob(pattern)):
            # run.py's own output tree, re-uploaded inside each artifact.
            if "workflow_logs" in path.parts:
                continue
            if path.name.startswith("report_data_"):
                continue
            found.append(path)
    return found


def mergeable(payload: Mapping[str, Any]) -> Tuple[bool, str]:
    """``(include?, reason)`` — shape only, never the test's name."""
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        return False, "no report sections"
    # The evals smoke run used to be dropped here. It is now merged like any
    # other report: collecting every result is worth more than avoiding the
    # duplicate rows. Be aware of what that costs — smoke and full measure the
    # same task and are identical in `kind`, `title`, `id` and `task_name`, so
    # the merged report carries two different scores for one task with nothing
    # to distinguish them, and both reach the database that way. Marking them
    # apart needs a sample-mode field on the eval block, which the schema does
    # not have. Re-adding the filter is a two-line change if that changes.
    return True, "included"


def _strip_server_url(command: str) -> str:
    """Drop the ``--server-url <value>`` pair and normalise whitespace."""
    if not command:
        return ""
    tokens = command.split()
    out: List[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--server-url":
            skip_next = True
            continue
        if token.startswith("--server-url="):
            continue
        out.append(token)
    return " ".join(out)


def deduplicate(
    sources: Sequence[SourceReport],
) -> Tuple[List[SourceReport], List[Tuple[SourceReport, SourceReport]]]:
    """Keep only the newest report per test.

    A test that fails its acceptance check still writes a report, so retrying it
    leaves two reports for one test in the same CI run — and merging both
    double-counts every section and lets the dead attempt's failure set the
    verdict even though the retry passed.

    Newest wins, ordered by job id (which rises monotonically across attempts)
    and then by ``generated_at``. Returns ``(kept, dropped)`` where each dropped
    entry is paired with the report that superseded it, so the caller can say
    what it discarded.
    """
    newest: Dict[str, SourceReport] = {}
    order: List[str] = []
    superseded: List[Tuple[SourceReport, SourceReport]] = []

    for source in sorted(sources, key=lambda s: (s.job_id, s.generated_at)):
        key = source.identity
        if key in newest:
            superseded.append((newest[key], source))
            newest[key] = source
        else:
            newest[key] = source
            order.append(key)

    kept = [newest[key] for key in order]
    kept.sort(key=lambda s: (s.job_id, s.generated_at))
    return kept, superseded


def load_test_reports(
    paths: Sequence[Path],
) -> Tuple[List[SourceReport], List[Tuple[Path, str]]]:
    """Parse and partition the candidates into (included, skipped)."""
    included: List[SourceReport] = []
    skipped: List[Tuple[Path, str]] = []

    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable report %s: %s", path.name, exc)
            skipped.append((path, "unreadable"))
            continue
        if not isinstance(payload, Mapping):
            skipped.append((path, "not a report object"))
            continue

        keep, reason = mergeable(payload)
        if not keep:
            logger.info("Skipping %s: %s", path.name, reason)
            skipped.append((path, reason))
            continue

        try:
            schema = ReportSchema.from_dict(payload)
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed report %s: %s", path.name, exc)
            skipped.append((path, f"malformed: {exc}"))
            continue

        included.append(SourceReport(path=path, schema=schema, payload=dict(payload)))

    included.sort(key=lambda source: source.job_id)
    return included, skipped


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def build_merged_schema(
    sources: Sequence[SourceReport],
    model: Optional[str] = None,
    target: Optional[str] = None,
    job_id: Optional[str] = None,
) -> ReportSchema:
    """One schema whose sections are every source's sections, in order.

    Sections are copied verbatim — the merge never inspects what a benchmark or
    an eval means, which is why an unfamiliar section kind costs nothing.
    """
    seed = dict(sources[0].schema.metadata) if sources else {}
    now = datetime.now(timezone.utc).replace(microsecond=0)

    metadata = {
        key: value
        for key, value in seed.items()
        # Per-test values that would be misleading on a merged report.
        if key not in ("run_command", "report_id", "generated_at", "workflow",
                       "runtime_model_spec_json")
    }
    model_name = str(metadata.get("model_name") or model or "")
    device = str(metadata.get("device") or "")

    metadata.update(
        {
            "model_name": model_name,
            "device": device,
            "workflow": MERGED_WORKFLOW,
            "generated_at": now.isoformat(),
            "report_id": _merged_report_id(model_name, device, now),
            # There is no single run.py invocation behind a merged report; the
            # per-test commands are listed under "Merged from" in the markdown.
            "run_command": _describe_merge(sources, model_name, target, job_id),
        }
    )

    sections = [block for source in sources for block in source.schema.sections]
    return ReportSchema(metadata=metadata, sections=sections)


def _merged_report_id(model_name: str, device: str, now: datetime) -> str:
    slug = (model_name or "unknown").replace("/", "__")
    device_slug = (device or "exabox").lower()
    return f"{slug}_{device_slug}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def _describe_merge(
    sources: Sequence[SourceReport],
    model_name: str,
    target: Optional[str],
    job_id: Optional[str],
) -> str:
    where = f" on {target}" if target else ""
    job = f" (job {job_id})" if job_id else ""
    return (
        f"merged from {len(sources)} exabox test report(s) for "
        f"{model_name or 'unknown model'}{where}{job}"
    )


def resolve_model_status(sources: Sequence[SourceReport]) -> str:
    """The model's status tier, as the per-test reports recorded it.

    This gates enforcement: the same ``accuracy_check`` counts as a failure for
    a FUNCTIONAL model and as unenforced for an EXPERIMENTAL one. Taking it
    from the sources keeps the merged verdict consistent with what each test
    already reported, rather than re-deriving it from a model spec this process
    does not load.
    """
    seen = {source.model_status for source in sources if source.model_status}
    if not seen:
        return ""
    if len(seen) > 1:
        # Reports can disagree if the model spec changed between CI attempts.
        # Returning "" makes acceptance_criteria_check enforce every tier and
        # every eval, matching the principle it states for itself: "an
        # absent/garbled status can never accidentally loosen acceptance."
        # Picking one of the values could let a stale, laxer status relax
        # enforcement for the whole merged report.
        logger.warning(
            "Source reports disagree on model_status (%s); enforcing all "
            "criteria rather than trusting either",
            ", ".join(sorted(seen)),
        )
        return ""
    return seen.pop()


def merge_reports(
    container_dir: Path,
    output_dir: Path,
    model: Optional[str] = None,
    target: Optional[str] = None,
    job_id: Optional[str] = None,
    missing_tests: Sequence[str] = (),
    keep_duplicates: bool = False,
) -> Tuple[Optional[GenerateResult], Dict[str, Any]]:
    """Merge everything under ``container_dir`` and render the report.

    Returns ``(result, stats)``. ``result`` is ``None`` only when no report
    could be rendered at all.
    """
    candidates = discover_test_reports(container_dir)
    sources, skipped = load_test_reports(candidates)

    superseded: List[Tuple[SourceReport, SourceReport]] = []
    if not keep_duplicates:
        sources, superseded = deduplicate(sources)
        for old, new in superseded:
            logger.info(
                "Superseded %s (job %d) by %s (job %d) — same test, later attempt",
                old.path.name, old.job_id, new.path.name, new.job_id,
            )

    logger.info(
        "Discovered %d report(s): %d merged, %d skipped, %d superseded",
        len(candidates),
        len(sources),
        len(skipped),
        len(superseded),
    )

    schema = build_merged_schema(sources, model=model, target=target, job_id=job_id)
    model_status = resolve_model_status(sources)

    accepted, blockers, categories = acceptance_criteria_check(
        schema, known_issues=None, model_status=model_status
    )

    # Re-apply waivers the individual tests already honoured. Without this a
    # known issue waived upstream would resurface as a blocker purely because
    # the model spec is not loaded here.
    previously_waived: Dict[str, str] = {}
    for source in sources:
        previously_waived.update(source.waived())
    if previously_waived:
        for category in categories:
            for key in list(category.blockers):
                if key in previously_waived:
                    category.waived[key] = category.blockers.pop(key)
        for key in list(blockers):
            if key in previously_waived:
                blockers.pop(key)
        accepted = not blockers
        logger.info("Re-applied %d waiver(s) from source reports", len(previously_waived))

    # A requested test that produced no report is invisible to the category
    # checks, which read a missing category as NA rather than a failure. Exabox
    # runs tests sequentially with fail-fast, so this is the normal shape of a
    # partial run — and without it a run where most tests never started reads
    # exactly like a clean one.
    if missing_tests:
        gap_blockers = task_failure_blockers(
            (test, 1, False) for test in missing_tests
        )
        blockers = {**blockers, **gap_blockers}
        accepted = False
        logger.warning(
            "%d requested test(s) produced no report: %s",
            len(missing_tests),
            ", ".join(missing_tests),
        )

    # Nothing merged means nothing was assessed. Left as accepted, a run in
    # which every test failed would be indistinguishable from a clean one.
    if not sources:
        accepted = False
        blockers = {
            **blockers,
            "exabox:no_reports": "No mergeable test reports were found for this run.",
        }

    schema.metadata.update(
        build_acceptance_export(accepted, blockers, categories, model_status)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = ReportGenerator().generate(schema, output_dir)

    stats = {
        "merged": len(sources),
        "skipped": len(skipped),
        "superseded": [
            {"dropped": old.path.name, "kept": new.path.name}
            for old, new in superseded
        ],
        "sections": len(schema.sections),
        "missing_tests": list(missing_tests),
        "model_status": model_status,
        "accepted": accepted,
        "blockers": len(blockers),
        "report_id": schema.metadata["report_id"],
        "merged_from": [source.path.name for source in sources],
        "skipped_reports": [
            {"file": path.name, "reason": reason} for path, reason in skipped
        ],
    }
    return result, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="exabox_merged_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--container-dir", required=True, type=Path,
                        help="Directory holding the downloaded per-test report artifacts")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write report_<id>.md and data/report_data_<id>.json")
    parser.add_argument("--model", default="", help="Model id, for the merge description")
    parser.add_argument("--target", default="", help="Target cluster, for the merge description")
    parser.add_argument("--job-id", default="",
                        help="Aggregating job id. When given, job-id-named copies are "
                             "written so the data collector can attribute the report.")
    parser.add_argument("--missing-test", action="append", default=[], dest="missing_tests",
                        metavar="TEST",
                        help="A requested test that produced no report; repeatable")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Merge every report found, including several attempts of the "
                             "same test. Off by default: retrying a test leaves two "
                             "reports, and merging both double-counts its sections and "
                             "lets the superseded attempt's failure set the verdict.")
    parser.add_argument("--stats-json", type=Path, default=None,
                        help="Also write the run stats to this path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.container_dir.exists():
        logger.warning("Container dir %s does not exist; treating as empty",
                       args.container_dir)
        args.container_dir.mkdir(parents=True, exist_ok=True)

    result, stats = merge_reports(
        container_dir=args.container_dir,
        output_dir=args.output_dir,
        model=args.model or None,
        target=args.target or None,
        job_id=args.job_id or None,
        missing_tests=args.missing_tests,
        keep_duplicates=args.keep_duplicates,
    )

    if result is None:
        logger.error("No report was rendered")
        return 1

    # The data collector keys a report to its CI job by the trailing
    # _<job_id> in the filename, and ignores anything else. The generator
    # names its output after report_id, so publish job-id-named copies too.
    if args.job_id:
        json_copy = args.output_dir / f"report_{args.job_id}.json"
        shutil.copyfile(result.json_path, json_copy)
        stats["json_path"] = str(json_copy)
    md_copy = args.output_dir / "release_report.md"
    shutil.copyfile(result.markdown_path, md_copy)
    stats["markdown_path"] = str(md_copy)

    payload = json.dumps(stats, indent=2)
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(payload)
    print(payload)
    logger.info(
        "Merged %d report(s) into %d section(s) -> %s",
        stats["merged"], stats["sections"], result.markdown_path,
    )
    return 0


if __name__ == "__main__":
    # Allow `python3 workflow_module/exabox_merged_report.py` as well as
    # `python3 -m workflow_module.exabox_merged_report`.
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
