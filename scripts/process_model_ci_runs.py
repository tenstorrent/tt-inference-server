#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Process GitHub CI runs for model testing.

This script uses the GitHub REST API to find "On nightly" job runs that succeeded
for each model defined in model_spec.py. It downloads artifacts, extracts performance
and accuracy data, and generates a summary report.

Usage:
    export GH_PAT='your_github_token'
    python3 scripts/process_model_ci_runs.py

Requirements:
    - Python 3.8+
    - Standard library only (no external dependencies)
    - GH_PAT environment variable with GitHub Personal Access Token
"""

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
from typing import Dict, List, Optional, Tuple, Set
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('process_model_ci_runs.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# GitHub API configuration
GITHUB_API = "https://api.github.com"
DEFAULT_OWNER = "tenstorrent"
DEFAULT_REPO = "tt-shield"
DEFAULT_WORKFLOW_FILE = "on-nightly.yml"
GITHUB_API_VERSION = "2022-11-28"


def _curl_debug_string(url: str, accept: Optional[str] = None) -> str:
    """Generate curl command string for debugging API calls."""
    hdrs = [
        "-H 'Authorization: Bearer $GH_PAT'",
        "-H 'User-Agent: process-model-ci-runs/1.0'",
        f"-H 'X-GitHub-Api-Version: {GITHUB_API_VERSION}'",
    ]
    if accept:
        hdrs.append(f"-H 'Accept: {accept}'")
    return f"curl -sS -L {' '.join(hdrs)} '{url}'"


def http_get(url: str, token: str, accept: Optional[str] = None, retry: int = 3) -> bytes:
    """Make HTTP GET request with retry logic."""
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "process-model-ci-runs/1.0",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if accept:
        headers["Accept"] = accept
    
    logger.info(f"HTTP GET {url}")
    logger.debug(f"Re-run with: {_curl_debug_string(url, accept)}")
    
    for attempt in range(retry):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=60) as resp:
                body = resp.read()
                logger.info(f"HTTP {resp.getcode()} {url} bytes={len(body)}")
                return body
        except HTTPError as e:
            logger.error(f"HTTPError {e.code} on {url} (attempt {attempt+1}/{retry})")
            logger.debug(f"Retry with: {_curl_debug_string(url, accept)}")
            if e.code in (429, 500, 502, 503, 504) and attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except URLError as e:
            logger.error(f"URLError on {url}: {e} (attempt {attempt+1}/{retry})")
            logger.debug(f"Retry with: {_curl_debug_string(url, accept)}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def http_json(url: str, token: str, retry: int = 3) -> dict:
    """Make HTTP GET request and parse JSON response."""
    data = http_get(url, token, accept="application/vnd.github+json", retry=retry)
    return json.loads(data.decode("utf-8"))


def check_authorization(owner: str, repo: str, token: str) -> bool:
    """Check GitHub token authorization and repository access."""
    logger.info(f"Checking authorization for repository: {owner}/{repo}")
    
    if not token:
        logger.error("GH_PAT environment variable not set")
        return False
    
    # Test repository access
    test_url = f"{GITHUB_API}/repos/{owner}/{repo}"
    try:
        repo_data = http_json(test_url, token)
        logger.info(f"Repository access confirmed: {repo_data.get('full_name')}")
        
        # Test workflows access
        workflows_url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows"
        workflows_data = http_json(workflows_url, token)
        workflow_count = len(workflows_data.get("workflows", []))
        logger.info(f"Workflows access confirmed: {workflow_count} workflows found")
        
        return True
    except Exception as e:
        logger.error(f"Authorization check failed: {e}")
        return False


def get_workflow(owner: str, repo: str, workflow_file: str, token: str) -> dict:
    """Get workflow information by filename."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_file}"
    logger.info(f"Getting workflow: {workflow_file}")
    return http_json(url, token)


def list_workflow_runs(workflow_id: int, owner: str, repo: str, token: str, per_page: int = 30) -> List[dict]:
    """List workflow runs for a specific workflow."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs?per_page={per_page}"
    logger.info(f"Listing workflow runs for workflow ID: {workflow_id}")
    data = http_json(url, token)
    runs = data.get("workflow_runs", [])
    logger.info(f"Found {len(runs)} workflow runs")
    return runs


def list_run_artifacts(run_id: int, owner: str, repo: str, token: str) -> List[dict]:
    """List artifacts for a specific workflow run."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    logger.info(f"Listing artifacts for run ID: {run_id}")
    data = http_json(url, token)
    artifacts = data.get("artifacts", [])
    logger.info(f"Found {len(artifacts)} artifacts for run {run_id}")
    return artifacts


def download_run_logs_zip(run_id: int, owner: str, repo: str, token: str) -> bytes:
    """Download workflow run logs as ZIP file."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    logger.info(f"Downloading logs for run ID: {run_id}")
    try:
        return http_get(url, token, accept="application/vnd.github+json")
    except HTTPError as e:
        if e.code in (406, 415):
            logger.info("Retrying run logs download without Accept header")
            return http_get(url, token, accept=None)
        raise


def download_artifact_zip(artifact_id: int, owner: str, repo: str, token: str) -> bytes:
    """Download workflow artifact as ZIP file."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
    logger.info(f"Downloading artifact ID: {artifact_id}")
    try:
        return http_get(url, token, accept="application/vnd.github+json")
    except HTTPError as e:
        if e.code in (406, 415):
            logger.info("Retrying artifact download without Accept header")
            return http_get(url, token, accept=None)
        if e.code in (400, 404, 410):
            logger.warning(f"Artifact download failed with HTTP {e.code}. Artifact may be expired.")
            try:
                logger.info("Retrying with Accept: application/octet-stream")
                return http_get(url, token, accept="application/octet-stream")
            except Exception:
                pass
        raise


def ensure_dir(path: Path) -> None:
    """Ensure directory exists, creating it if necessary."""
    if path.exists():
        logger.debug(f"Directory already exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def extract_zip_to_dir(zip_bytes: bytes, out_dir: Path) -> None:
    """Extract ZIP bytes to directory."""
    ensure_dir(out_dir)
    logger.info(f"Extracting ZIP to: {out_dir}")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)
    logger.info(f"Extracted files to: {out_dir}")


def find_commits_from_logs(logs_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract tt_metal_commit and vllm_commit from build-inference-server logs."""
    logger.info(f"Scanning logs for commits in: {logs_dir}")
    
    tt_metal_commit = None
    vllm_commit = None
    
    # Regex patterns to match commit arguments
    commit_re_tt = re.compile(r"--tt-metal-commit\"\s*\"([^\"]+)\"")
    commit_re_vllm = re.compile(r"--vllm-commit\"\s*\"([^\"]+)\"")
    
    # Search all .txt files, prioritizing build-inference-server job
    txt_files = sorted(logs_dir.rglob("*.txt"))
    
    for fpath in txt_files:
        name_lower = fpath.name.lower()
        logger.debug(f"Scanning log file: {fpath}")
        
        try:
            content = fpath.read_text(errors="ignore")
        except Exception as e:
            logger.debug(f"Could not read {fpath}: {e}")
            continue
        
        # Extract commits
        if tt_metal_commit is None:
            m1 = commit_re_tt.search(content)
            if m1:
                tt_metal_commit = m1.group(1)
                logger.info(f"Found tt_metal_commit: {tt_metal_commit} in {fpath}")
        
        if vllm_commit is None:
            m2 = commit_re_vllm.search(content)
            if m2:
                vllm_commit = m2.group(1)
                logger.info(f"Found vllm_commit: {vllm_commit} in {fpath}")
        
        if tt_metal_commit and vllm_commit:
            break
    
    return tt_metal_commit, vllm_commit


def parse_perf_status(report_data: dict) -> str:
    """
    Parse performance status from report data.
    
    Returns highest target achieved: target > complete > functional > experimental
    Pass condition: ttft_check != 3 AND tput_user_check != 3 AND tput_check != 3
    """
    try:
        summaries = report_data.get("benchmarks_summary", [])
        if not summaries:
            return "experimental"
        
        target_checks = summaries[0].get("target_checks", {})
        
        def passes(checks: dict) -> bool:
            """Check if all performance checks pass (none equal 3)."""
            if not isinstance(checks, dict):
                return False
            ttft_check = checks.get("ttft_check")
            tput_user_check = checks.get("tput_user_check")
            tput_check = checks.get("tput_check")
            return all(x is not None and x != 3 for x in (ttft_check, tput_user_check, tput_check))
        
        # Check targets in order of highest to lowest
        if passes(target_checks.get("target", {})):
            return "target"
        if passes(target_checks.get("complete", {})):
            return "complete"
        if passes(target_checks.get("functional", {})):
            return "functional"
        return "experimental"
    
    except Exception as e:
        logger.debug(f"Error parsing perf status: {e}")
        return "experimental"


def parse_accuracy_status(report_data: dict) -> bool:
    """
    Parse accuracy status from report data.
    
    Returns True if all accuracy checks pass (accuracy_check != 3).
    """
    try:
        evals = report_data.get("evals", [])
        if not evals:
            return False
        
        for eval_item in evals:
            if eval_item.get("accuracy_check") == 3:
                return False
        
        return True
    
    except Exception as e:
        logger.debug(f"Error parsing accuracy status: {e}")
        return False


def latest_json_by_mtime(dir_path: Path, pattern: str) -> Optional[Path]:
    """Find the most recent JSON file matching pattern by modification time."""
    logger.debug(f"Looking for pattern '{pattern}' in: {dir_path}")
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_model_spec_json(run_specs_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Load model specification JSON from run_specs directory."""
    spec_file = latest_json_by_mtime(run_specs_dir, "*.json")
    if not spec_file:
        logger.debug(f"No JSON files found in: {run_specs_dir}")
        return None, None
    
    try:
        logger.info(f"Reading model spec JSON: {spec_file}")
        data = json.loads(spec_file.read_text())
        model_id = data.get("model_id")
        return data, model_id
    except Exception as e:
        logger.error(f"Error loading model spec from {spec_file}: {e}")
        return None, None


def process_artifact_dir(artifact_dir: Path) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    """
    Process artifact directory to extract model spec and report data.
    
    Returns:
        Tuple of (model_spec_json, report_data_json, model_id)
    """
    logger.info(f"Processing artifact directory: {artifact_dir}")
    
    # Load model spec from run_specs directory
    run_specs_dir = artifact_dir / "run_specs"
    model_spec_json, model_id = load_model_spec_json(run_specs_dir)
    if not model_id:
        logger.warning(f"No model_id found in: {run_specs_dir}")
        return None, None, None
    
    # Load report data from reports_output directory
    reports_root = artifact_dir / "reports_output"
    report_data_json = None
    
    if reports_root.exists():
        # Search workflow subdirectories (release, benchmarks, evals, etc.)
        for workflow_dir in reports_root.iterdir():
            if not workflow_dir.is_dir():
                continue
            
            data_dir = workflow_dir / "data"
            if not data_dir.is_dir():
                continue
            
            # Look for report_data_<model_id>_*.json
            report_file = latest_json_by_mtime(data_dir, f"report_data_{model_id}_*.json")
            if not report_file:
                # Fallback: any report_data_*.json
                report_file = latest_json_by_mtime(data_dir, "report_data_*.json")
            
            if report_file:
                try:
                    logger.info(f"Reading report data JSON: {report_file}")
                    report_data_json = json.loads(report_file.read_text())
                    break
                except Exception as e:
                    logger.error(f"Error loading report data from {report_file}: {e}")
    
    if not report_data_json:
        logger.warning(f"No report data found for model_id: {model_id}")
    
    return model_spec_json, report_data_json, model_id


def format_datetime(dt_str: str) -> str:
    """Convert ISO datetime string to YYYY-MM-DD_HH-MM-SS format."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        # Fallback: simple string replacement
        return dt_str.replace(":", "-").replace("T", "_").replace("Z", "")


def get_models_from_workflow_yml() -> Set[str]:
    """
    Extract model names from the on-nightly.yml workflow file.
    
    Since the .github directory doesn't exist in this repo, we'll extract
    models from the model_spec.py file instead.
    """
    logger.info("Extracting model names from model specifications")
    
    # Import the model specs from the workflows module
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from workflows.model_spec import MODEL_SPECS
        
        models = set()
        for model_id, spec in MODEL_SPECS.items():
            model_name = spec.model_name
            models.add(model_name)
        
        logger.info(f"Found {len(models)} unique models in model specifications")
        for model in sorted(models):
            logger.debug(f"  - {model}")
        
        return models
    
    except Exception as e:
        logger.error(f"Error loading model specifications: {e}")
        # Fallback: hardcoded list based on the workflow YAML provided
        models = {
            "gemma-3-1b-it", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
            "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "gemma-3-4b-it",
            "Llama-3.2-11B-Vision-Instruct", "Qwen2.5-7B-Instruct", "gemma-3-27b-it",
            "Llama-3.3-70B-Instruct", "Llama-3.2-90B-Vision-Instruct", "QwQ-32B",
            "Qwen2.5-72B-Instruct", "Qwen3-8B", "Qwen3-32B", "stable-diffusion-xl-base-1.0",
            "stable-diffusion-3.5-large", "distil-whisper/distil-large-v3", "microsoft/resnet-50"
        }
        logger.info(f"Using fallback model list with {len(models)} models")
        return models


def main():
    """Main function to process GitHub CI runs for model testing."""
    parser = argparse.ArgumentParser(
        description="Process GitHub CI runs for model testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process last 30 nightly runs
    export GH_PAT='your_token'
    python3 scripts/process_model_ci_runs.py

    # Process last 50 runs from specific repo
    python3 scripts/process_model_ci_runs.py --owner tenstorrent --repo tt-shield --max-runs 50

Environment Variables:
    GH_PAT: GitHub Personal Access Token (required)
        """
    )
    
    parser.add_argument("--owner", default=DEFAULT_OWNER,
                       help=f"GitHub repository owner (default: {DEFAULT_OWNER})")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                       help=f"GitHub repository name (default: {DEFAULT_REPO})")
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE,
                       help=f"Workflow filename (default: {DEFAULT_WORKFLOW_FILE})")
    parser.add_argument("--max-runs", type=int, default=30,
                       help="Maximum number of workflow runs to process (default: 30)")
    parser.add_argument("--out-root", type=str, default=".",
                       help="Output root directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Step 0: Check environment and authorization
    token = os.getenv("GH_PAT")
    if not token:
        logger.error("GH_PAT environment variable not set")
        logger.error("Please set your GitHub Personal Access Token:")
        logger.error("export GH_PAT='your_github_token'")
        return 1
    
    if not check_authorization(args.owner, args.repo, token):
        logger.error("Authorization check failed")
        return 1
    
    # Step 1: Setup output directory
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    logger.info(f"Output root directory: {out_root}")
    
    # Step 2: Get workflow and list runs
    try:
        workflow = get_workflow(args.owner, args.repo, args.workflow_file, token)
        workflow_id = workflow.get("id")
        if not workflow_id:
            logger.error("Could not resolve workflow ID")
            return 1
        
        logger.info(f"Found workflow '{workflow.get('name')}' with ID: {workflow_id}")
    except Exception as e:
        logger.error(f"Error getting workflow: {e}")
        return 1
    
    # Step 3: List workflow runs
    try:
        runs = list_workflow_runs(workflow_id, args.owner, args.repo, token, per_page=args.max_runs)
        if not runs:
            logger.warning("No workflow runs found")
            return 0
    except Exception as e:
        logger.error(f"Error listing workflow runs: {e}")
        return 1
    
    # Step 4: Process each run
    passing_dict: Dict[str, List[dict]] = {}
    all_run_timestamps: List[str] = []
    
    for i, run in enumerate(runs, 1):
        run_id = run.get("id")
        run_number = run.get("run_number")
        run_started_at = run.get("run_started_at") or run.get("created_at") or run.get("updated_at")
        
        logger.info(f"Processing run {i}/{len(runs)}: #{run_number} (ID: {run_id})")
        
        # Format timestamp
        if run_started_at:
            run_ts_str = format_datetime(run_started_at)
            all_run_timestamps.append(run_ts_str)
        else:
            run_ts_str = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create run directory
        run_dir_name = f"On_nightly_{run_number}"
        run_out_dir = out_root / run_dir_name
        ensure_dir(run_out_dir)
        
        # Download and extract run logs to find commits
        test_tt_metal_commit: Optional[str] = None
        test_vllm_commit: Optional[str] = None
        
        try:
            logs_zip = download_run_logs_zip(run_id, args.owner, args.repo, token)
            logs_dir = run_out_dir / "logs"
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            extract_zip_to_dir(logs_zip, logs_dir)
            test_tt_metal_commit, test_vllm_commit = find_commits_from_logs(logs_dir)
        except Exception as e:
            logger.warning(f"Could not download/process logs for run {run_id}: {e}")
        
        # List and download workflow log artifacts
        try:
            artifacts = list_run_artifacts(run_id, args.owner, args.repo, token)
        except Exception as e:
            logger.warning(f"Could not list artifacts for run {run_id}: {e}")
            continue
        
        # Process workflow_logs artifacts
        for artifact in artifacts:
            name = artifact.get("name", "")
            if not name.startswith("workflow_logs_"):
                continue
            
            artifact_id = artifact.get("id")
            if not artifact_id:
                continue
            
            logger.info(f"Processing artifact: {name}")
            
            try:
                # Download and extract artifact
                artifact_zip = download_artifact_zip(artifact_id, args.owner, args.repo, token)
                artifact_dir = run_out_dir / name
                if artifact_dir.exists():
                    shutil.rmtree(artifact_dir)
                extract_zip_to_dir(artifact_zip, artifact_dir)
                
                # Process extracted directory
                model_spec_json, report_data_json, model_id = process_artifact_dir(artifact_dir)
                
                if not model_id or not model_spec_json or not report_data_json:
                    logger.debug(f"Incomplete data for artifact {name}, skipping")
                    continue
                
                # Parse performance and accuracy status
                perf_status = parse_perf_status(report_data_json)
                accuracy_status = parse_accuracy_status(report_data_json)
                
                logger.info(f"Model {model_id}: perf_status={perf_status}, accuracy_status={accuracy_status}")
                
                # Store data for passing jobs
                entry = {
                    "job_run_datetimestamp": run_ts_str,
                    "test_tt_metal_commit": test_tt_metal_commit,
                    "test_vllm_commit": test_vllm_commit,
                    "perf_status": perf_status,
                    "accuracy_status": accuracy_status,
                    "model_spec_json": model_spec_json,
                }
                
                passing_dict.setdefault(model_id, []).append(entry)
                
            except Exception as e:
                logger.warning(f"Error processing artifact {name}: {e}")
                continue
    
    # Step 5: Generate summary output
    if all_run_timestamps:
        earliest = min(all_run_timestamps)
        latest = max(all_run_timestamps)
    else:
        now_str = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        earliest = latest = now_str
    
    # Save complete summary with model_spec_json
    summary_filename = f"models_ci_on_nightly_pass_fail_summary_{earliest}_to_{latest}.json"
    summary_path = out_root / summary_filename
    
    logger.info(f"Writing complete summary to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(passing_dict, f, indent=2)
    logger.info(f"Wrote summary to: {summary_path}")
    
    # Step 6: Generate and print latest passing values (without model_spec_json)
    latest_compact: Dict[str, dict] = {}
    
    for model_id, entries in passing_dict.items():
        # Sort by timestamp and take the latest
        entries_sorted = sorted(entries, key=lambda e: e.get("job_run_datetimestamp", ""))
        if entries_sorted:
            latest_entry = entries_sorted[-1]
            latest_compact[model_id] = {
                "test_tt_metal_commit": latest_entry.get("test_tt_metal_commit"),
                "test_vllm_commit": latest_entry.get("test_vllm_commit"),
                "perf_status": latest_entry.get("perf_status"),
                "accuracy_status": latest_entry.get("accuracy_status"),
            }
    
    # Print compact summary
    logger.info(f"Found data for {len(latest_compact)} models")
    print("\nLatest passing values per model:")
    print(json.dumps(latest_compact, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
