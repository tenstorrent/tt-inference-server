#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Training-workflow driver.

Runs in run.py's interpreter (NOT inside the container): the forge inference
server is already up as a sibling ``ServerCommand``; this launcher acts as the
HTTP client that submits one LoRA fine-tuning job, waits for it to finish,
pulls its per-step loss from ``GET /v1/jobs/{id}/metrics``, grades the loss
trajectory against a checked-in expectation and writes a ``report_data_*.json``
so tt-shield's existing artifact/collect_data steps pick it up.

The report is run through ``report_module.acceptance_criteria`` so the loss
records (``kind: "spec_tests"``) are graded by ``_check_spec_tests`` for free:
the ``.md`` gains an ``### Acceptance Criteria`` section and the JSON gains a
top-level ``acceptance_criteria`` key — the same verdict/Slack/dashboard flow
perf and evals already use, with no tt-shield change.

Exit code is non-zero if the job failed OR the loss checks failed, so the CI
step gates on it. By default acceptance is *advisory* (surfaced but not the
gate) because the checked-in goldens are still placeholders
(``TODO(regenerate-on-hardware)``); pass ``--enforce-acceptance`` once real
goldens land to make the acceptance verdict drive the exit code.

Usage (flags mirror the other engine launchers)::

    python launchers/run_training_test.py \
        --model meta-llama/Llama-3.1-8B --workflow training_tests --device p150 \
        --service-port 8000 --runtime-model-spec-json /tmp/spec.json \
        --output-dir workflow_logs/reports_output/training_tests \
        --expected-config reference_config/training/llama_3_1_8b_sst2_p150.yaml \
        --docker-server
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# launchers/<this file> -> parent is launchers/, parent.parent is the repo root.
# Unlike the venv-reexec launchers, this driver runs directly in run.py's
# interpreter as a subprocess, so the repo root is not guaranteed to be on
# sys.path; add it so ``workflows`` / ``report_module`` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("tt_training_launcher")

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_SUCCESS_STATUSES = {"completed"}


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--service-port", required=True)
    parser.add_argument("--runtime-model-spec-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-config", required=True)
    parser.add_argument("--jwt-secret", default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--docker-server", action="store_true")
    # Make the acceptance verdict (which always enforces spec_tests) drive the
    # exit code. Off by default: the checked-in goldens are placeholders, so we
    # emit + surface the acceptance verdict without gating CI on it. Flip this
    # on once reference_config/training/*.yaml holds real P150 losses.
    parser.add_argument("--enforce-acceptance", action="store_true")
    # Bounds (seconds). Server readiness for an 8B model + first-run tensor
    # cache can be very slow; the job itself compiles then trains.
    parser.add_argument("--health-timeout", type=float, default=3600.0)
    parser.add_argument("--job-timeout", type=float, default=5400.0)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    args, _ = parser.parse_known_args(argv)
    if args.workflow != "training_tests":
        parser.error(
            "run_training_test.py requires --workflow training_tests "
            f"(got {args.workflow})."
        )
    return args


def _base_url(args: argparse.Namespace) -> str:
    if args.server_url:
        return args.server_url.rstrip("/")
    return f"http://127.0.0.1:{args.service_port}"


def _auth_headers(jwt_secret: Optional[str] = None) -> Dict[str, str]:
    """Build the auth headers the forge/media fine-tuning endpoints expect.

    The training/fine-tuning endpoints live on the media server and require two
    things (see ``tt-media-server/security/``):

    * ``Authorization: Bearer $API_KEY`` — a literal string compare against
      ``$API_KEY`` (default ``"your-secret-key"``); it does NOT decode a JWT
      (that is the vLLM auth model). ``$NO_AUTH`` disables the check
      server-side, in which case no auth header is needed.
    * A non-empty org header (``get_org_id``); its name is configurable via
      ``$ORG_ID_HEADER`` (default ``X-TT-Organization``). The value is only
      used to scope jobs to a tenant, so any non-empty id works for tests.
    """
    headers: Dict[str, str] = {}

    org_header = os.getenv("ORG_ID_HEADER", "X-TT-Organization")
    org_id = os.getenv("TT_ORG_ID", "tenstorrent")
    headers[org_header] = org_id

    if os.getenv("NO_AUTH", "").lower() in ("1", "true", "yes"):
        return headers

    api_key = os.getenv("API_KEY", "your-secret-key")
    headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _wait_for_health(session, base_url: str, headers, timeout: float) -> bool:
    import requests

    deadline = time.time() + timeout
    health_url = f"{base_url}/health"
    while time.time() < deadline:
        try:
            resp = session.get(health_url, headers=headers, timeout=30)
            if resp.status_code == 200:
                logger.info("Server healthy at %s", health_url)
                return True
            logger.info("Health %s -> %s; waiting...", health_url, resp.status_code)
        except requests.exceptions.RequestException as exc:
            logger.info("Health check not ready (%s); waiting...", exc)
        time.sleep(10)
    logger.error("Server did not become healthy within %ss", timeout)
    return False


def _build_request_body(device: str, request_overrides: Dict[str, Any]) -> Dict[str, Any]:
    body = {"device_type": device}
    body.update(request_overrides)
    # device_type must match the server device; the expectation file should not
    # override it, but guard anyway.
    body["device_type"] = device
    return body


def _submit_job(session, base_url, headers, body) -> Optional[str]:
    import requests

    url = f"{base_url}/v1/jobs"
    # The endpoint returns 405 until the model is ready; retry a few times in
    # case health passed but warmup is still finishing.
    for attempt in range(1, 11):
        try:
            resp = session.post(url, headers=headers, json=body, timeout=120)
        except requests.exceptions.RequestException as exc:
            logger.warning("Submit attempt %d failed to connect: %s", attempt, exc)
            time.sleep(15)
            continue
        if resp.status_code == 201:
            data = resp.json()
            job_id = data.get("id") or data.get("job_id") or data.get("task_id")
            logger.info("Submitted training job: %s", job_id)
            return job_id
        if resp.status_code == 405:
            logger.info("Model not ready (405); retry %d/10 ...", attempt)
            time.sleep(15)
            continue
        logger.error("Submit failed (%s): %s", resp.status_code, resp.text[:500])
        return None
    logger.error("Model never became ready for job submission.")
    return None


def _poll_until_terminal(
    session, base_url, headers, job_id, timeout, interval
) -> Optional[str]:
    import requests

    url = f"{base_url}/v1/jobs/{job_id}"
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        try:
            resp = session.get(url, headers=headers, timeout=60)
            if resp.status_code == 200:
                status = str(resp.json().get("status", "")).lower()
                if status != last_status:
                    logger.info("Job %s status: %s", job_id, status)
                    last_status = status
                if status in _TERMINAL_STATUSES:
                    return status
            else:
                logger.info("Job poll %s -> %s", url, resp.status_code)
        except requests.exceptions.RequestException as exc:
            logger.info("Job poll error (%s); retrying...", exc)
        time.sleep(interval)
    logger.error("Job %s did not reach a terminal state within %ss", job_id, timeout)
    return None


def _fetch_metrics(session, base_url, headers, job_id) -> List[Dict[str, Any]]:
    import requests

    url = f"{base_url}/v1/jobs/{job_id}/metrics"
    try:
        resp = session.get(url, headers=headers, timeout=60)
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch metrics: %s", exc)
        return []
    if resp.status_code != 200:
        logger.error("Metrics fetch failed (%s): %s", resp.status_code, resp.text[:500])
        return []
    payload = resp.json()
    if isinstance(payload, dict):
        return payload.get("metrics") or payload.get("data") or []
    if isinstance(payload, list):
        return payload
    return []


def _read_acceptance_inputs(
    spec_json_path: Optional[str],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract ``(model_status, known_issues)`` from the runtime model spec JSON.

    The launcher receives the same
    ``{"runtime_model_spec": …, "runtime_config": …}`` document the engine
    writes: ``status`` sits at the top of ``runtime_model_spec`` (a serialized
    enum name, e.g. ``"EXPERIMENTAL"``) and per-device ``known_issues`` waivers
    live under its ``device_model_spec``. Both feed
    ``acceptance_criteria_check`` exactly as the engine's
    ``execution.apply_acceptance_criteria`` supplies them.

    Falls back to the dev-spec default (``EXPERIMENTAL`` / no waivers) when the
    path is missing or unreadable. ``spec_tests`` are enforced regardless of
    status, so this fallback never softens the gate.
    """
    fallback_status = "EXPERIMENTAL"
    known_issues: List[Dict[str, Any]] = []
    if not spec_json_path:
        return fallback_status, known_issues
    try:
        data = json.loads(Path(spec_json_path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read runtime model spec %s: %s", spec_json_path, exc)
        return fallback_status, known_issues
    spec = data.get("runtime_model_spec") if isinstance(data, dict) else None
    if not isinstance(spec, dict):
        return fallback_status, known_issues
    model_status = spec.get("status") or fallback_status
    device_spec = spec.get("device_model_spec")
    if isinstance(device_spec, dict) and isinstance(
        device_spec.get("known_issues"), list
    ):
        known_issues = device_spec["known_issues"]
    return model_status, known_issues


def _write_report(
    output_dir,
    model,
    device,
    records,
    extra_metadata,
    *,
    model_status=None,
    known_issues=None,
) -> bool:
    """Render the training report, folding in an acceptance verdict.

    Running the schema through ``acceptance_criteria_check`` grades the
    ``spec_tests`` loss records via ``_check_spec_tests`` and — by stashing the
    export in ``metadata`` before ``generate_report`` — makes the generator emit
    both the ``### Acceptance Criteria`` markdown section and the top-level
    ``acceptance_criteria`` JSON key that tt-shield already harvests. Returns
    whether acceptance passed so the caller can gate on it.
    """
    from report_module.acceptance_criteria import (
        KIND_SPEC_TESTS,
        acceptance_criteria_check,
        build_acceptance_export,
    )
    from report_module.generator import generate_report
    from report_module.schema import Block, ReportSchema

    metadata = {
        "model_name": model,
        "device": device,
        "workflow": "training_tests",
        "report_id": f"{model.replace('/', '__')}_{device}_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.utcnow().isoformat(),
    }
    metadata.update(extra_metadata)

    # Each record becomes its own titled Block. `_check_spec_tests` grades a
    # *block-level* status, but `ReportSchema.from_records` would collapse the
    # flat records into a single wrapper block with no top-level status —
    # resolving (via legacy fallback) to a spurious blocking FAIL regardless of
    # the records. One block per record instead gives correct per-test grading
    # and per-test blocker keys (`spec.spec_tests:<test_name>`). Each such block
    # renders to an empty table (all its columns are hidden) and is dropped, so
    # the layout is unchanged: the generator injects its 🧪 Test Results summary
    # from the same records.
    sections = [
        Block(
            kind=str(record.get("kind") or KIND_SPEC_TESTS),
            data=dict(record),
            title=str(record["test_name"]) if record.get("test_name") else None,
        )
        for record in records
    ]
    schema = ReportSchema(metadata=metadata, sections=sections)
    accepted, blockers, categories = acceptance_criteria_check(
        schema, known_issues=known_issues, model_status=model_status
    )
    schema.metadata.update(
        build_acceptance_export(accepted, blockers, categories, model_status)
    )
    generate_report(schema, output_dir)
    return accepted


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:])

    import requests
    import yaml

    from workflows.training.loss_check import evaluate, parse_config

    expected_path = Path(args.expected_config)
    if not expected_path.is_file():
        logger.error("Expected-config not found: %s", expected_path)
        return 1
    config = parse_config(yaml.safe_load(expected_path.read_text()))

    model_status, known_issues = _read_acceptance_inputs(args.runtime_model_spec_json)

    base_url = _base_url(args)
    headers = {"Content-Type": "application/json", **_auth_headers(args.jwt_secret)}
    session = requests.Session()

    output_dir = Path(args.output_dir)

    if not _wait_for_health(session, base_url, headers, args.health_timeout):
        _write_report(
            output_dir,
            args.model,
            args.device,
            [
                {
                    "kind": "spec_tests",
                    "model": args.model,
                    "device": args.device,
                    "test_name": "server_healthy",
                    "status": "fail",
                    "attempts": 1,
                    "elapsed_seconds": 0.0,
                    "description": "server never became healthy",
                }
            ],
            {"verdict": "FAIL"},
            model_status=model_status,
            known_issues=known_issues,
        )
        return 1

    body = _build_request_body(args.device, config.request)
    logger.info("Submitting training job with body: %s", json.dumps(body))
    job_id = _submit_job(session, base_url, headers, body)
    if not job_id:
        _write_report(
            output_dir,
            args.model,
            args.device,
            [
                {
                    "kind": "spec_tests",
                    "model": args.model,
                    "device": args.device,
                    "test_name": "job_submitted",
                    "status": "fail",
                    "attempts": 1,
                    "elapsed_seconds": 0.0,
                    "description": "failed to submit training job",
                }
            ],
            {"verdict": "FAIL"},
            model_status=model_status,
            known_issues=known_issues,
        )
        return 1

    final_status = _poll_until_terminal(
        session, base_url, headers, job_id, args.job_timeout, args.poll_interval
    )
    metrics = _fetch_metrics(session, base_url, headers, job_id)

    result = evaluate(metrics, config, model=args.model, device=args.device)

    job_succeeded = final_status in _SUCCESS_STATUSES
    if not job_succeeded:
        result.records.append(
            {
                "kind": "spec_tests",
                "model": args.model,
                "device": args.device,
                "test_name": "job_completed",
                "status": "fail",
                "attempts": 1,
                "elapsed_seconds": 0.0,
                "description": f"job terminal status={final_status}",
            }
        )

    passed = result.passed and job_succeeded
    accepted = _write_report(
        output_dir,
        args.model,
        args.device,
        result.records,
        {"verdict": "PASS" if passed else "FAIL", "summary": result.summary},
        model_status=model_status,
        known_issues=known_issues,
    )

    # Acceptance always enforces spec_tests, so `accepted` already reflects the
    # loss checks and the appended job-failure record. Gate on it only when the
    # goldens are trusted; otherwise keep the status-quo loss/job gate and let
    # the (advisory) acceptance verdict ride along in the report for dashboards.
    gate_passed = accepted if args.enforce_acceptance else passed
    logger.info(
        "Training check summary: %s (job_status=%s, acceptance=%s, mode=%s)",
        result.summary,
        final_status,
        "PASS" if accepted else "FAIL",
        "enforcing" if args.enforce_acceptance else "advisory",
    )
    if not args.enforce_acceptance and not accepted:
        logger.warning(
            "Acceptance verdict is FAIL but --enforce-acceptance is off; "
            "surfacing it in the report without gating CI (goldens are still "
            "placeholders)."
        )
    if gate_passed:
        logger.info("✅ Training workflow passed.")
        return 0
    logger.error("⛔ Training workflow failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
