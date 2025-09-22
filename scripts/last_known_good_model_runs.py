#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import io
import re
import json
import time
import shutil
import zipfile
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('last_known_good_model_runs.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


GITHUB_API = "https://api.github.com"
DEFAULT_OWNER = "tenstorrent"
DEFAULT_REPO = "tt-shield"
DEFAULT_WORKFLOW_FILE = "on-nightly.yml"  # .github/workflows/on-nightly.yml


def http_get(url: str, token: str, accept: Optional[str] = None, retry: int = 3) -> bytes:
    logger.info(f"Making HTTP GET request to: {url}")
    headers = {"Authorization": f"Bearer {token[:10]}...", "User-Agent": "last-known-good-model-runs/1.0"}
    if accept:
        headers["Accept"] = accept
    for attempt in range(retry):
        try:
            req = Request(url, headers=headers, method="GET")
            logger.debug(f"HTTP request attempt {attempt + 1}/{retry}")
            with urlopen(req, timeout=60) as resp:
                logger.info(f"HTTP request successful: {resp.getcode()} {resp.reason}")
                return resp.read()
        except HTTPError as e:
            logger.warning(f"HTTP error on attempt {attempt + 1}: {e.code} {e.reason}")
            if e.code in (429, 500, 502, 503, 504) and attempt < retry - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            raise
        except URLError as e:
            logger.warning(f"URL error on attempt {attempt + 1}: {e}")
            if attempt < retry - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            raise


def http_json(url: str, token: str, retry: int = 3) -> dict:
    data = http_get(url, token, accept="application/vnd.github+json", retry=retry)
    return json.loads(data.decode("utf-8"))


def list_workflows(owner: str, repo: str, token: str) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows"
    data = http_json(url, token)
    return data.get("workflows", [])


def get_workflow(owner: str, repo: str, workflow_file: str, token: str) -> dict:
    # Try direct resolution by filename
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_file}"
    try:
        return http_json(url, token)
    except HTTPError as e:
        if e.code != 404:
            raise
    # Fallback: list workflows and match by name or path suffix
    workflows = list_workflows(owner, repo, token)
    for wf in workflows:
        path = wf.get("path", "")
        name = wf.get("name", "")
        if path.endswith(f"/{workflow_file}"):
            return wf
        if name.lower() == "on nightly":
            return wf
    raise HTTPError(url, 404, f"Workflow {workflow_file} not found", hdrs=None, fp=None)


def list_workflow_runs(workflow_id: int, owner: str, repo: str, token: str, per_page: int = 30) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs?per_page={per_page}"
    data = http_json(url, token)
    return data.get("workflow_runs", [])


def list_repo_runs(owner: str, repo: str, token: str, per_page: int = 100) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs?per_page={per_page}"
    data = http_json(url, token)
    return data.get("workflow_runs", [])


def list_run_artifacts(run_id: int, owner: str, repo: str, token: str) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    data = http_json(url, token)
    return data.get("artifacts", [])


def download_run_logs_zip(run_id: int, owner: str, repo: str, token: str) -> bytes:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    return http_get(url, token, accept="application/zip")


def download_artifact_zip(artifact_id: int, owner: str, repo: str, token: str) -> bytes:
    # Endpoint requires "/zip"
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
    return http_get(url, token, accept="application/zip")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_zip_to_dir(zip_bytes: bytes, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)


def find_commits_from_logs(logs_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    tt_metal_commit = None
    vllm_commit = None

    # Search all .txt files; prioritize those likely from build job
    txt_files = sorted(logs_dir.rglob("*.txt"))
    def maybe_extract(line: str) -> Tuple[Optional[str], Optional[str]]:
        tt = None
        vl = None
        m_tt = re.search(r"--tt[-_]?metal[-_]?commit\s*\"?([^\"\s]+)\"?", line)
        if m_tt:
            tt = m_tt.group(1)
        m_vl = re.search(r"--vllm[-_]?commit\s*\"?([^\"\s]+)\"?", line)
        if m_vl:
            vl = m_vl.group(1)
        return tt, vl

    # Two passes: first filter build files, then others
    prioritized = [p for p in txt_files if "build-inference-server" in p.name.lower() or "build_inference_server" in p.name.lower()]
    ordered = prioritized + [p for p in txt_files if p not in prioritized]
    for fpath in ordered:
        try:
            with fpath.open("r", errors="ignore") as fh:
                for line in fh:
                    if tt_metal_commit is None or vllm_commit is None:
                        tt, vl = maybe_extract(line)
                        if tt_metal_commit is None and tt:
                            tt_metal_commit = tt
                        if vllm_commit is None and vl:
                            vllm_commit = vl
                    else:
                        break
        except Exception:
            continue
        if tt_metal_commit and vllm_commit:
            break

    return tt_metal_commit, vllm_commit


def parse_perf_status(report_data: dict) -> str:
    # Determine highest target achieved among target, complete, functional
    # Pass condition for a level: all checks != 3
    try:
        summaries = report_data.get("benchmarks_summary", [])
        if not summaries:
            return "experimental"
        target_checks = summaries[0].get("target_checks", {})
        def passes(checks: dict) -> bool:
            if not isinstance(checks, dict):
                return False
            ttft_check = checks.get("ttft_check")
            tput_user_check = checks.get("tput_user_check")
            tput_check = checks.get("tput_check")
            return all(x is not None and x != 3 for x in (ttft_check, tput_user_check, tput_check))

        # Order of highest to lowest
        if passes(target_checks.get("target", {})):
            return "target"
        if passes(target_checks.get("complete", {})):
            return "complete"
        if passes(target_checks.get("functional", {})):
            return "functional"
        return "experimental"
    except Exception:
        return "experimental"


def parse_accuracy_status(report_data: dict) -> bool:
    try:
        evals = report_data.get("evals", [])
        if not evals:
            return False
        for e in evals:
            if e.get("accuracy_check") == 3:
                return False
        return True
    except Exception:
        return False


def latest_json_by_mtime(dir_path: Path, pattern: str) -> Optional[Path]:
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_model_spec_json(run_specs_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
    spec_file = latest_json_by_mtime(run_specs_dir, "*.json")
    if not spec_file:
        return None, None
    try:
        data = json.loads(spec_file.read_text())
    except Exception:
        return None, None
    model_id = data.get("model_id")
    return data, model_id


def process_artifact_dir(artifact_dir: Path) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    # Returns (model_spec_json, report_data_json, model_id)
    run_specs_dir = artifact_dir / "run_specs"
    model_spec_json, model_id = load_model_spec_json(run_specs_dir)
    if not model_id:
        return None, None, None
    # reports_output/<workflow>/data/report_data_<model_id>_*.json
    reports_root = artifact_dir / "reports_output"
    report_data_json = None
    if reports_root.exists():
        # workflow subdirectory can vary (release, benchmarks, evals)
        for workflow_dir in reports_root.iterdir():
            data_dir = workflow_dir / "data"
            if data_dir.is_dir():
                report_file = latest_json_by_mtime(data_dir, f"report_data_{model_id}_*.json")
                if not report_file:
                    # fallback: any report_data_*.json
                    report_file = latest_json_by_mtime(data_dir, "report_data_*.json")
                if report_file:
                    try:
                        report_data_json = json.loads(report_file.read_text())
                        break
                    except Exception:
                        pass
    return model_spec_json, report_data_json, model_id


def format_dt(dt_str: str) -> str:
    # Convert ISO to YYYY-MM-DD_HH-MM-SS
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        return dt_str.replace(":", "-").replace("T", "_").replace("Z", "")


def main():
    parser = argparse.ArgumentParser(description="Find last known good model runs from On nightly CI")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--out-root", type=str, default=".")
    args = parser.parse_args()

    logger.info("Starting last_known_good_model_runs.py script")
    logger.info(f"Arguments: owner={args.owner}, repo={args.repo}, workflow_file={args.workflow_file}, max_runs={args.max_runs}")

    token = os.getenv("GH_PAT")
    if not token:
        logger.error("GH_PAT environment variable not set")
        raise ValueError("GH_PAT not set in environment")

    logger.info(f"GH_PAT token found (length: {len(token)} characters)")
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    logger.info(f"Output directory: {out_root}")

    # 1) Resolve workflow and list recent runs
    runs: List[dict] = []
    workflow_name_filter = "On nightly"

    logger.info(f"Searching for workflow: {args.workflow_file}")
    try:
        wf = get_workflow(args.owner, args.repo, args.workflow_file, token)
        workflow_id = wf.get("id")
        logger.info(f"Found workflow ID: {workflow_id}")
        if workflow_id:
            logger.info(f"Fetching up to {args.max_runs} workflow runs")
            runs = list_workflow_runs(workflow_id, args.owner, args.repo, token, per_page=args.max_runs)
            logger.info(f"Found {len(runs)} workflow runs")
    except HTTPError as e:
        logger.warning(f"Workflow-specific API call failed: {e.code} {e.reason}, falling back to repo-level runs")
        # Fall through to repo-level runs
        pass

    if not runs:
        logger.info("No runs found from workflow API, trying repo-level API")
        # Fallback: list repo runs and filter by workflow name
        logger.info(f"Fetching up to {max(100, args.max_runs)} runs from repo API")
        repo_runs = list_repo_runs(args.owner, args.repo, token, per_page=max(100, args.max_runs))
        logger.info(f"Found {len(repo_runs)} total runs in repo")

        on_nightly_runs = [r for r in repo_runs if str(r.get("name", "")).lower() == workflow_name_filter.lower()]
        logger.info(f"Filtered to {len(on_nightly_runs)} 'On nightly' runs")
        on_nightly_runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        runs = on_nightly_runs[: args.max_runs]
        logger.info(f"Using {len(runs)} most recent runs")
    passing_dict: Dict[str, List[dict]] = {}
    all_run_timestamps: List[str] = []

    logger.info(f"Processing {len(runs)} workflow runs...")
    for i, run in enumerate(runs):
        run_id = run.get("id")
        run_number = run.get("run_number")
        run_started_at = run.get("run_started_at") or run.get("created_at") or run.get("updated_at")

        logger.info(f"Processing run {i+1}/{len(runs)}: ID={run_id}, Number={run_number}")

        if run_started_at:
            run_ts_str = format_dt(run_started_at)
            all_run_timestamps.append(run_ts_str)
            logger.debug(f"Run timestamp: {run_ts_str}")
        else:
            run_ts_str = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            logger.warning(f"No timestamp found for run {run_id}, using current time")

        # Create run dir e.g. On_nightly_236
        run_dir_name = f"On_nightly_{run_number}"
        run_out_dir = out_root / run_dir_name
        ensure_dir(run_out_dir)
        logger.debug(f"Created output directory: {run_out_dir}")

        # Download run logs and parse build-inference-server args
        test_tt_metal_commit: Optional[str] = None
        test_vllm_commit: Optional[str] = None
        logger.info(f"Downloading logs for run {run_id}")
        try:
            logs_zip = download_run_logs_zip(run_id, args.owner, args.repo, token)
            logs_dir = run_out_dir / "logs"
            logger.debug(f"Extracting logs to: {logs_dir}")
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            extract_zip_to_dir(logs_zip, logs_dir)
            test_tt_metal_commit, test_vllm_commit = find_commits_from_logs(logs_dir)
            if test_tt_metal_commit:
                logger.info(f"Found tt-metal commit: {test_tt_metal_commit}")
            if test_vllm_commit:
                logger.info(f"Found vllm commit: {test_vllm_commit}")
        except Exception as e:
            logger.warning(f"Failed to download/parse logs for run {run_id}: {e}")

        # List and download artifacts matching workflow logs
        logger.info(f"Listing artifacts for run {run_id}")
        try:
            artifacts = list_run_artifacts(run_id, args.owner, args.repo, token)
            logger.info(f"Found {len(artifacts)} artifacts")
        except Exception as e:
            logger.warning(f"Failed to list artifacts for run {run_id}: {e}")
            artifacts = []

        workflow_artifacts_processed = 0
        for artifact in artifacts:
            name = artifact.get("name", "")
            logger.debug(f"Processing artifact: {name}")
            if not name.startswith("workflow_logs_"):
                logger.debug(f"Skipping non-workflow artifact: {name}")
                continue
            artifact_id = artifact.get("id")
            if not artifact_id:
                logger.warning(f"Artifact {name} has no ID, skipping")
                continue
            logger.info(f"Downloading artifact: {name} (ID: {artifact_id})")
            try:
                z = download_artifact_zip(artifact_id, args.owner, args.repo, token)
                # Extract under run directory in folder named as artifact name
                art_dir = run_out_dir / name
                logger.debug(f"Extracting artifact to: {art_dir}")
                if art_dir.exists():
                    shutil.rmtree(art_dir)
                extract_zip_to_dir(z, art_dir)
                workflow_artifacts_processed += 1
            except Exception as e:
                logger.warning(f"Failed to download artifact {name}: {e}")
                continue

            # Process extracted directory
            logger.debug(f"Processing artifact directory: {art_dir}")
            model_spec_json, report_data_json, model_id = process_artifact_dir(art_dir)
            if not model_id or not model_spec_json or not report_data_json:
                logger.warning(f"Failed to extract data from artifact directory: {art_dir}")
                continue

            logger.info(f"Processing model: {model_id}")
            perf_status = parse_perf_status(report_data_json)
            accuracy_status = parse_accuracy_status(report_data_json)

            logger.info(f"Model {model_id}: perf_status={perf_status}, accuracy_status={accuracy_status}")

            # Mark pass/fail and append only passing
            is_pass = (perf_status != "experimental") and accuracy_status
            if is_pass:
                logger.info(f"✅ Model {model_id} PASSED")
                entry = {
                    "job_run_datetimestamp": run_ts_str,
                    "test_tt_metal_commit": test_tt_metal_commit,
                    "test_vllm_commit": test_vllm_commit,
                    "perf_status": perf_status,
                    "accuracy_status": accuracy_status,
                    "model_spec_json": model_spec_json,
                }
                passing_dict.setdefault(model_id, []).append(entry)
            else:
                logger.info(f"❌ Model {model_id} FAILED")

        logger.info(f"Processed {workflow_artifacts_processed} workflow artifacts for run {run_id}")

    # 3) Serialize passing_dict to JSON file
    logger.info("Generating summary output...")
    if all_run_timestamps:
        earliest = min(all_run_timestamps)
        latest = max(all_run_timestamps)
        logger.info(f"Date range: {earliest} to {latest}")
    else:
        now_s = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        earliest = latest = now_s
        logger.warning("No timestamps found, using current time")

    summary_name = f"models_ci_on_nightly_pass_fail_summary_{earliest}_to_{latest}.json"
    summary_path = out_root / summary_name
    logger.info(f"Writing summary file: {summary_path}")
    summary_path.write_text(json.dumps(passing_dict, indent=2))

    # 4) Print latest passing values per model_id (without model_spec_json)
    logger.info("Processing latest passing results...")
    latest_compact: Dict[str, dict] = {}
    for model_id, entries in passing_dict.items():
        # choose entry with max job_run_datetimestamp
        entries_sorted = sorted(entries, key=lambda e: e.get("job_run_datetimestamp", ""))
        chosen = entries_sorted[-1]
        latest_compact[model_id] = {
            "test_tt_metal_commit": chosen.get("test_tt_metal_commit"),
            "test_vllm_commit": chosen.get("test_vllm_commit"),
            "perf_status": chosen.get("perf_status"),
            "accuracy_status": chosen.get("accuracy_status"),
        }
        logger.info(f"Latest result for {model_id}: perf={chosen.get('perf_status')}, accuracy={chosen.get('accuracy_status')}")

    # Print as JSON
    logger.info(f"Found {len(latest_compact)} passing models")
    print(json.dumps(latest_compact, indent=2))
    logger.info("Script completed successfully")


if __name__ == "__main__":
    sys.exit(main())
