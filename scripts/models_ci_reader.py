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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


GITHUB_API = "https://api.github.com"
DEFAULT_OWNER = "tenstorrent"
DEFAULT_REPO = "tt-shield"
DEFAULT_WORKFLOW_FILE = "on-nightly.yml"  # .github/workflows/on-nightly.yml


def http_get(url: str, token: str, accept: Optional[str] = None, retry: int = 3) -> bytes:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "models-ci-reader/1.0"}
    if accept:
        headers["Accept"] = accept
    for attempt in range(retry):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except URLError:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def http_json(url: str, token: str, retry: int = 3) -> dict:
    data = http_get(url, token, accept="application/vnd.github+json", retry=retry)
    return json.loads(data.decode("utf-8"))


def get_workflow(owner: str, repo: str, workflow_file: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_file}"
    return http_json(url, token)


def list_workflow_runs(workflow_id: int, owner: str, repo: str, token: str, per_page: int = 30) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs?per_page={per_page}"
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

    # Search all .txt files, prefer ones that look like build-inference-server job
    txt_files = sorted(logs_dir.rglob("*.txt"))
    commit_re_tt = re.compile(r"--tt-metal-commit\"\s*\"([^\"]+)\"")
    commit_re_vllm = re.compile(r"--vllm-commit\"\s*\"([^\"]+)\"")

    for fpath in txt_files:
        name_lower = fpath.name.lower()
        if "build-inference-server" not in name_lower and "build_inference_server" not in name_lower:
            # Still scan everything, but prioritize build job first
            pass
        try:
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue
        if tt_metal_commit is None:
            m1 = commit_re_tt.search(content)
            if m1:
                tt_metal_commit = m1.group(1)
        if vllm_commit is None:
            m2 = commit_re_vllm.search(content)
            if m2:
                vllm_commit = m2.group(1)
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
    parser = argparse.ArgumentParser(description="Read On nightly CI results and summarize passing models")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--out-root", type=str, default=".")
    args = parser.parse_args()

    token = os.getenv("GH_PAT")
    assert token, "GH_PAT not set in environment"

    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    # 1) Resolve workflow and list recent runs
    wf = get_workflow(args.owner, args.repo, args.workflow_file, token)
    workflow_id = wf.get("id")
    assert workflow_id, "Could not resolve workflow id"

    runs = list_workflow_runs(workflow_id, args.owner, args.repo, token, per_page=args.max_runs)
    passing_dict: Dict[str, List[dict]] = {}
    all_run_timestamps: List[str] = []

    for run in runs:
        run_id = run.get("id")
        run_number = run.get("run_number")
        run_started_at = run.get("run_started_at") or run.get("created_at") or run.get("updated_at")
        if run_started_at:
            run_ts_str = format_dt(run_started_at)
            all_run_timestamps.append(run_ts_str)
        else:
            run_ts_str = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        # Create run dir e.g. On_nightly_236
        run_dir_name = f"On_nightly_{run_number}"
        run_out_dir = out_root / run_dir_name
        ensure_dir(run_out_dir)

        # Download run logs and parse build-inference-server args
        test_tt_metal_commit: Optional[str] = None
        test_vllm_commit: Optional[str] = None
        try:
            logs_zip = download_run_logs_zip(run_id, args.owner, args.repo, token)
            logs_dir = run_out_dir / "logs"
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            extract_zip_to_dir(logs_zip, logs_dir)
            test_tt_metal_commit, test_vllm_commit = find_commits_from_logs(logs_dir)
        except Exception:
            pass

        # List and download artifacts matching workflow logs
        try:
            artifacts = list_run_artifacts(run_id, args.owner, args.repo, token)
        except Exception:
            artifacts = []

        for artifact in artifacts:
            name = artifact.get("name", "")
            if not name.startswith("workflow_logs_"):
                continue
            artifact_id = artifact.get("id")
            if not artifact_id:
                continue
            try:
                z = download_artifact_zip(artifact_id, args.owner, args.repo, token)
                # Extract under run directory in folder named as artifact name
                art_dir = run_out_dir / name
                if art_dir.exists():
                    shutil.rmtree(art_dir)
                extract_zip_to_dir(z, art_dir)
            except Exception:
                continue

            # Process extracted directory
            model_spec_json, report_data_json, model_id = process_artifact_dir(art_dir)
            if not model_id or not model_spec_json or not report_data_json:
                continue

            perf_status = parse_perf_status(report_data_json)
            accuracy_status = parse_accuracy_status(report_data_json)

            # Mark pass/fail and append only passing
            is_pass = (perf_status != "experimental") and accuracy_status
            if is_pass:
                entry = {
                    "job_run_datetimestamp": run_ts_str,
                    "test_tt_metal_commit": test_tt_metal_commit,
                    "test_vllm_commit": test_vllm_commit,
                    "perf_status": perf_status,
                    "accuracy_status": accuracy_status,
                    "model_spec_json": model_spec_json,
                }
                passing_dict.setdefault(model_id, []).append(entry)

    # 3) Serialize passing_dict to JSON file
    if all_run_timestamps:
        earliest = min(all_run_timestamps)
        latest = max(all_run_timestamps)
    else:
        now_s = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        earliest = latest = now_s
    summary_name = f"models_ci_on_nightly_pass_fail_summary_{earliest}_to_{latest}.json"
    summary_path = out_root / summary_name
    summary_path.write_text(json.dumps(passing_dict, indent=2))

    # 4) Print latest passing values per model_id (without model_spec_json)
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

    # Print as JSON
    print(json.dumps(latest_compact, indent=2))


if __name__ == "__main__":
    sys.exit(main())


