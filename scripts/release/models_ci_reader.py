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
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError

from workflow_logs_parser import parse_workflow_logs_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('models_ci_reader.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_OWNER = "tenstorrent"
DEFAULT_REPO = "tt-shield"
DEFAULT_WORKFLOW_FILE = "on-nightly.yml"  # .github/workflows/on-nightly.yml
GITHUB_API_VERSION = "2022-11-28"


def _curl_debug_string(url: str, accept: Optional[str]) -> str:
    hdrs = [
        "-H 'Authorization: Bearer $GH_PAT'",
        "-H 'User-Agent: models-ci-reader/1.0'",
        f"-H 'X-GitHub-Api-Version: {GITHUB_API_VERSION}'",
    ]
    if accept:
        hdrs.append(f"-H 'Accept: {accept}'")
    return f"curl -sS -L {' '.join(hdrs)} '{url}'"


class _StripAuthOnRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Clone headers except Authorization
        filtered_headers = {k: v for k, v in req.header_items() if k.lower() != 'authorization'}
        return Request(newurl, headers=filtered_headers, method=req.get_method())


def http_get(url: str, token: str, accept: Optional[str] = None, retry: int = 3, timeout: int = 60, auth_scheme: str = "Bearer", strip_auth_on_redirect: bool = False) -> bytes:
    headers = {
        "Authorization": f"{auth_scheme} {token}",
        "User-Agent": "models-ci-reader/1.0",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if accept:
        headers["Accept"] = accept
    logger.debug(f"HTTP GET {url}")
    logger.debug(f"Re-run with: {_curl_debug_string(url, accept)}")
    for attempt in range(retry):
        try:
            req = Request(url, headers=headers, method="GET")
            if strip_auth_on_redirect:
                opener = build_opener(_StripAuthOnRedirect)
                resp_ctx = opener.open(req, timeout=timeout)
            else:
                resp_ctx = urlopen(req, timeout=timeout)
            with resp_ctx as resp:
                body = resp.read()
                logger.debug(f"HTTP {resp.getcode()} {url} bytes={len(body)}")
                return body
        except HTTPError as e:
            logger.debug(f"HTTPError {e.code} on {url} (attempt {attempt+1}/{retry})")
            logger.debug(f"Retry with: {_curl_debug_string(url, accept)}")
            if e.code in (429, 500, 502, 503, 504) and attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except URLError as e:
            logger.debug(f"URLError on {url}: {e} (attempt {attempt+1}/{retry})")
            logger.debug(f"Retry with: {_curl_debug_string(url, accept)}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def http_json(url: str, token: str, retry: int = 3) -> dict:
    data = http_get(url, token, accept="application/vnd.github+json", retry=retry, timeout=60, auth_scheme="Bearer")
    return json.loads(data.decode("utf-8"))


def get_workflow(owner: str, repo: str, workflow_file: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_file}"
    return http_json(url, token)


def list_workflow_runs(workflow_id: int, owner: str, repo: str, token: str, per_page: int = 30) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs?per_page={per_page}"
    data = http_json(url, token)
    return data.get("workflow_runs", [])


def get_workflow_run(run_id: int, owner: str, repo: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}"
    return http_json(url, token)


def list_run_artifacts(run_id: int, owner: str, repo: str, token: str) -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    data = http_json(url, token)
    return data.get("artifacts", [])


def list_run_jobs(run_id: int, owner: str, repo: str, token: str) -> List[dict]:
    """List all jobs for a workflow run with pagination support.
    
    This returns all jobs including those from reusable workflows.
    Jobs from reusable workflows have different run_id values which can be used
    to download their logs separately.
    """
    all_jobs = []
    page = 1
    per_page = 100
    
    while True:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/jobs?per_page={per_page}&page={page}"
        data = http_json(url, token)
        jobs = data.get("jobs", [])
        
        if not jobs:
            break
            
        all_jobs.extend(jobs)
        
        # Check if there are more pages
        if len(jobs) < per_page:
            break
            
        page += 1
    
    return all_jobs


def download_job_logs(job_id: int, owner: str, repo: str, token: str) -> bytes:
    """Download logs for a specific job using REST API.
    
    Equivalent to: curl -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer $GH_PAT" <url> -o file.txt
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }
    
    # Handle redirect manually (curl -L behavior)
    req = Request(url, headers=headers)
    
    class RedirectHandler(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            # Strip Authorization header on redirect (like curl does for cross-domain)
            new_headers = {k: v for k, v in req.headers.items() if k != 'Authorization'}
            return Request(newurl, headers=new_headers)
    
    opener = build_opener(RedirectHandler)
    with opener.open(req, timeout=300) as response:
        return response.read()


def download_ci_job_logs(jobs: List[dict], logs_dir: Path, owner: str, repo: str, token: str) -> Tuple[int, int]:
    """Download individual job logs using REST API.
    
    Args:
        jobs: List of job dicts from list_run_jobs()
        logs_dir: Directory to save log files
        owner: Repository owner
        repo: Repository name
        token: GitHub API token
    
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    logger.info(f"Downloading logs for {len(jobs)} jobs...")
    successful_downloads = 0
    failed_downloads = 0
    
    for job in jobs:
        job_id = job.get("id")
        job_name = job.get("name", f"job_{job_id}")
        
        if not job_id:
            failed_downloads += 1
            continue
        
        # Sanitize job name for filesystem
        safe_job_name = sanitize_filename(job_name)
        log_filename = f"{safe_job_name}_{job_id}.txt"
        log_path = logs_dir / log_filename
        
        try:
            job_logs = download_job_logs(job_id, owner, repo, token)
            log_path.write_bytes(job_logs)
            logger.info(f"  ✓ {log_filename}")
            successful_downloads += 1
        except Exception:
            failed_downloads += 1
            continue
    
    logger.info(f"Downloaded {successful_downloads}/{len(jobs)} job logs")
    return successful_downloads, failed_downloads


def download_run_logs_zip(run_id: int, owner: str, repo: str, token: str) -> bytes:
    """Download consolidated workflow run logs as a ZIP file.
    
    This downloads logs for all jobs that are direct children of the workflow run.
    Jobs from reusable workflows are not included and must be obtained from artifacts.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    try:
        # GitHub API returns 302 redirect to ZIP file, need to strip auth on redirect
        return http_get(url, token, accept="application/vnd.github+json", timeout=300, auth_scheme="Bearer", strip_auth_on_redirect=True)
    except HTTPError as e:
        if e.code in (406, 415):
            logger.info("Retrying run logs download without Accept header due to HTTP error")
            return http_get(url, token, accept=None, timeout=300, auth_scheme="Bearer", strip_auth_on_redirect=True)
        raise


def download_artifact_zip(artifact_ref, owner: str, repo: str, token: str) -> bytes:
    """Download an artifact ZIP by id or by full archive_download_url.
    
    GitHub's artifact download API returns a 302 redirect to the actual download URL.
    The redirected URL should not include the Authorization header.
    """
    # Allow passing in archive_download_url directly
    if isinstance(artifact_ref, str) and artifact_ref.startswith("http"):
        # GitHub's archive_download_url also requires the same redirect handling
        # Don't use http_get directly - use the same logic as below
        url = artifact_ref
    else:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts/{artifact_ref}/zip"

    # Step 1: Make initial request to GitHub API to get redirect URL
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "models-ci-reader/1.0",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    
    logger.debug(f"HTTP GET {url}")
    
    # Create a custom redirect handler that doesn't follow redirects
    class NoRedirectHandler(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            # Don't follow any redirects - we'll handle them manually
            return None
    
    try:
        req = Request(url, headers=headers, method="GET")
        # Build opener with no redirect handling
        opener = build_opener(NoRedirectHandler)
        
        with opener.open(req, timeout=300) as resp:
            # This should not happen - GitHub should return 302
            if resp.getcode() == 200:
                body = resp.read()
                logger.debug(f"HTTP 200 {url} bytes={len(body)}")
                return body
            else:
                raise HTTPError(url, resp.getcode(), f"Unexpected response code: {resp.getcode()}", resp.headers, None)
                
    except HTTPError as e:
        if e.code == 302:
            # This is expected - GitHub returns 302 with Location header
            redirect_url = e.headers.get('Location')
            if not redirect_url:
                raise RuntimeError(f"GitHub API returned 302 but no Location header for {url}")
            
            logger.debug(f"Following redirect")
            
            # Step 2: Download from redirect URL WITHOUT Authorization header
            redirect_headers = {
                "User-Agent": "models-ci-reader/1.0",
            }
            
            redirect_req = Request(redirect_url, headers=redirect_headers, method="GET")
            with urlopen(redirect_req, timeout=300) as redirect_resp:
                body = redirect_resp.read()
                logger.debug(f"Downloaded {len(body)} bytes")
                return body
                
        elif e.code == 401:
            logger.error(f"Authentication failed for artifact download: {url}")
            logger.error("This may indicate:")
            logger.error("1. Token lacks 'repo' scope or Actions permissions")
            logger.error("2. Artifact has expired (artifacts expire after 90 days by default)")
            logger.error("3. Repository access permissions have changed")
            raise
        elif e.code in (400, 404, 410):
            logger.warning(
                f"Artifact download failed with HTTP {e.code} for {url}. This artifact may be expired or unavailable."
            )
            raise
        else:
            logger.error(f"HTTPError {e.code} on {url}")
            raise
            
    except URLError as e:
        logger.error(f"URLError on {url}: {e}")
        raise


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    invalid_chars = ['/', ':', '\\', '*', '?', '"', '<', '>', '|']
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_zip_to_dir(zip_bytes: bytes, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_list = zf.namelist()
        if len(file_list) == 0:
            logger.warning("ZIP file is empty!")
        zf.extractall(out_dir)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences to simplify regex matching."""
    try:
        ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        return ansi_re.sub("", text)
    except Exception:
        return text


def _parse_commits_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract tt-metal and vllm commit values from a block of text using multiple patterns."""
    text = _strip_ansi(text)

    patterns_tt = [
        r"--tt-metal-commit\"\s*\"([0-9a-fA-F]+)\"",        # "--tt-metal-commit" "<hash>"
        r"--tt-metal-commit[=\s]+\"?([0-9a-fA-F]{6,})\"?",    # --tt-metal-commit <hash> or =<hash>
        r"\btt-metal-commit:\s*([0-9a-zA-Z._-]+)\b",          # Inputs: tt-metal-commit: <value>
        r"\bTT_METAL_COMMIT\s*[:=]\s*\"?([0-9a-fA-F]{6,})\"?",
    ]

    patterns_vllm = [
        r"--vllm-commit\"\s*\"([0-9a-fA-F]+)\"",           # "--vllm-commit" "<hash>"
        r"--vllm-commit[=\s]+\"?([0-9a-fA-F]{3,})\"?",       # --vllm-commit <hash> or short sha
        r"\bvllm-commit:\s*([0-9a-zA-Z._-]+)\b",              # Inputs: vllm-commit: <value>
        r"\bVLLM_COMMIT\s*[:=]\s*\"?([0-9a-fA-F]{3,})\"?",
    ]

    tt_metal_commit: Optional[str] = None
    vllm_commit: Optional[str] = None

    for patt in patterns_tt:
        m = re.search(patt, text)
        if m:
            tt_metal_commit = m.group(1)
            break

    for patt in patterns_vllm:
        m = re.search(patt, text)
        if m:
            vllm_commit = m.group(1)
            break

    return tt_metal_commit, vllm_commit


def find_commits_from_logs(logs_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Find commits from the build log in the extracted run logs directory.

    This prefers a top-level file matching '*_build-inference-server*.txt'.
    Falls back to scanning other .txt files if not found.
    """
    logger.info(f"Scanning logs for commits in: {logs_dir}")

    # Prefer the consolidated build log file
    build_logs: List[Path] = []
    try:
        build_logs = sorted(logs_dir.glob("*_build-inference-server*.txt"))
    except Exception:
        build_logs = []

    # If not found, expand search
    search_files: List[Path]
    if build_logs:
        search_files = build_logs
    else:
        search_files = sorted(logs_dir.rglob("*.txt"))

    tt_metal_commit: Optional[str] = None
    vllm_commit: Optional[str] = None

    for fpath in search_files:
        try:
            logger.debug(f"Reading log file: {fpath}")
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue

        tt, vl = _parse_commits_from_text(content)
        if tt and not tt_metal_commit:
            tt_metal_commit = tt
        if vl and not vllm_commit:
            vllm_commit = vl
        if tt_metal_commit and vllm_commit:
            break

    return tt_metal_commit, vllm_commit


def _strip_timestamp_prefix(line: str) -> str:
    """Strip GitHub Actions timestamp prefix from log line."""
    timestamp_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+"
    return re.sub(timestamp_pattern, "", line)


def parse_tt_smi_from_logs(logs_dir: Path) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """Extract tt-smi output, firmware_bundle, and kmd_version from log files.
    
    Returns:
        Tuple of (tt_smi_output dict, firmware_bundle string, kmd_version string)
    """
    logger.info(f"Scanning logs for tt-smi output in: {logs_dir}")
    
    search_files = sorted(logs_dir.rglob("*.txt"))
    
    for fpath in search_files:
        try:
            logger.debug(f"Reading log file for tt-smi: {fpath}")
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue
        
        content = _strip_ansi(content)
        lines = content.splitlines()
        
        # Find tt-smi-metal -s command followed by JSON output
        for i, line in enumerate(lines):
            if "tt-smi-metal -s" in line:
                # Look for the JSON block starting with '{'
                json_lines = []
                found_start = False
                brace_count = 0
                
                for j in range(i + 1, len(lines)):
                    stripped_line = _strip_timestamp_prefix(lines[j])
                    
                    if not found_start:
                        if stripped_line.strip() == "{":
                            found_start = True
                            json_lines.append(stripped_line)
                            brace_count = 1
                    else:
                        json_lines.append(stripped_line)
                        brace_count += stripped_line.count("{")
                        brace_count -= stripped_line.count("}")
                        
                        if brace_count == 0:
                            break
                
                if json_lines and found_start and brace_count == 0:
                    try:
                        json_text = "\n".join(json_lines)
                        tt_smi_output = json.loads(json_text)
                        
                        # Validate that we have a proper tt-smi output structure
                        if not isinstance(tt_smi_output, dict):
                            logger.debug(f"Invalid tt-smi output: not a dict in {fpath.name}")
                            continue
                        
                        # Extract firmware_bundle from device_info
                        firmware_bundle: Optional[str] = None
                        device_info = tt_smi_output.get("device_info", [])
                        if device_info and isinstance(device_info, list) and len(device_info) > 0:
                            first_device = device_info[0]
                            if isinstance(first_device, dict):
                                firmwares = first_device.get("firmwares", {})
                                if isinstance(firmwares, dict):
                                    firmware_bundle = firmwares.get("fw_bundle_version")
                        
                        # Extract kmd_version from host_info.Driver (e.g., "TT-KMD 1.33")
                        kmd_version: Optional[str] = None
                        host_info = tt_smi_output.get("host_info", {})
                        if isinstance(host_info, dict):
                            driver_str = host_info.get("Driver")
                            if driver_str and isinstance(driver_str, str):
                                kmd_version = driver_str

                        # remove tt_smi_output.device_info
                        tt_smi_output.pop("device_info", None)

                        logger.info(f"Successfully extracted tt-smi output from {fpath.name}")
                        logger.info(f"Firmware bundle: {firmware_bundle}")
                        logger.info(f"KMD version: {kmd_version}")
                        return tt_smi_output, firmware_bundle, kmd_version
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse tt-smi JSON from {fpath.name}: {e}")
                        continue
    
    logger.warning("Could not find valid tt-smi output in logs")
    return None, None, None


def parse_runner_names(logs_dir: Path) -> Dict[str, str]:
    """Return mapping of job directory name -> runner name parsed from system.txt files."""
    result: Dict[str, str] = {}
    try:
        for sys_txt in logs_dir.rglob("system.txt"):
            try:
                text = sys_txt.read_text(errors="ignore")
            except Exception:
                continue
            text = _strip_ansi(text)
            m = re.search(r"Job is about to start running on the runner:\s*(.+)", text)
            if m:
                runner_name = m.group(1).strip()
                job_dir_name = sys_txt.parent.name
                result[job_dir_name] = runner_name
    except Exception:
        pass
    return result


def parse_built_docker_images_from_logs(logs_dir: Path) -> List[str]:
    """Extract successfully built docker images from build-inference-server logs.
    
    Looks for lines following '✅ Successfully built and pushed images:' in build logs.
    
    Returns:
        List of built docker image strings
    """
    logger.info(f"Scanning logs for built docker images in: {logs_dir}")
    
    # Look for build-inference-server logs
    build_logs = list(logs_dir.glob("*build-inference-server*.txt"))
    
    built_images: List[str] = []
    
    for fpath in build_logs:
        try:
            logger.debug(f"Reading build log file: {fpath}")
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue
        
        content = _strip_ansi(content)
        lines = content.splitlines()
        
        # Look for "Successfully built and pushed images:" marker
        found_marker = False
        for i, line in enumerate(lines):
            stripped = _strip_timestamp_prefix(line).strip()
            
            if "Successfully built and pushed images:" in stripped:
                found_marker = True
                # Parse subsequent lines until we hit a line that's not an image
                for j in range(i + 1, len(lines)):
                    next_stripped = _strip_timestamp_prefix(lines[j]).strip()
                    # Check if line looks like a docker image (starts with registry URL)
                    if next_stripped.startswith("ghcr.io/") or next_stripped.startswith("docker.io/"):
                        built_images.append(next_stripped)
                        logger.debug(f"Found built image: {next_stripped}")
                    else:
                        # Stop when we hit a non-image line
                        break
                break
    
    if built_images:
        logger.info(f"Found {len(built_images)} successfully built docker images")
    else:
        logger.warning("Could not find built docker images in build logs")
    
    return built_images


def parse_docker_image_from_logs(logs_dir: Path) -> Optional[str]:
    """Extract docker image from workflow inputs in log files.
    
    Looks for the 'image:' line in the Inputs section of GitHub Actions logs.
    Validates against built images from build-inference-server logs.
    
    Returns:
        Docker image string or None if not found
    """
    logger.info(f"Scanning logs for docker image in: {logs_dir}")
    
    search_files = sorted(logs_dir.rglob("*.txt"))
    
    docker_image_from_inputs: Optional[str] = None
    
    for fpath in search_files:
        try:
            logger.debug(f"Reading log file for docker image: {fpath}")
            content = fpath.read_text(errors="ignore")
        except Exception:
            continue
        
        content = _strip_ansi(content)
        lines = content.splitlines()
        
        # Look for the Inputs section followed by 'image:' line
        in_inputs_section = False
        for i, line in enumerate(lines):
            stripped = _strip_timestamp_prefix(line).strip()
            
            # Detect start of Inputs section
            if "##[group] Inputs" in line or "##[group]Inputs" in line:
                in_inputs_section = True
                continue
            
            # Detect end of Inputs section
            if in_inputs_section and "##[endgroup]" in line:
                in_inputs_section = False
                continue
            
            # Look for 'image:' line within Inputs section
            if in_inputs_section:
                image_match = re.match(r'\s*image:\s*(.+)', stripped)
                if image_match:
                    docker_image_from_inputs = image_match.group(1).strip()
                    logger.info(f"Found docker image in inputs from {fpath.name}: {docker_image_from_inputs}")
                    break
        
        if docker_image_from_inputs:
            break
    
    if not docker_image_from_inputs:
        logger.warning("Could not find docker image in workflow inputs")
        return None
    
    # Validate against built images
    built_images = parse_built_docker_images_from_logs(logs_dir)
    if built_images:
        if docker_image_from_inputs in built_images:
            logger.info(f"✅ Docker image from inputs validated against build logs: {docker_image_from_inputs}")
        else:
            logger.warning(f"⚠️  Docker image from inputs not found in build logs")
            logger.warning(f"   Input image: {docker_image_from_inputs}")
            logger.warning(f"   Built images: {built_images}")
            # Still return the input image even if validation fails
    
    return docker_image_from_inputs


def _parse_single_ci_job_log(log_file: Path, jobs_ci_metadata: Optional[List[dict]] = None) -> dict:
    """Extract system information from a single CI job log file.
    
    Args:
        log_file: Path to individual job log file
        jobs_ci_metadata: Optional list of job metadata dicts from jobs_ci_metadata.json
        
    Returns:
        Dict with keys: tt_smi_output, firmware_bundle, kmd_version, docker_image, runner_name, job_id, job_conclusion
    """
    result = {
        "tt_smi_output": None,
        "firmware_bundle": None,
        "kmd_version": None,
        "docker_image": None,
        "runner_name": None,
        "job_id": None,
        "job_conclusion": None,
    }
    
    # Extract job_id from filename (format: {safe_job_name}_{job_id}.txt)
    filename_stem = log_file.stem
    if "_" in filename_stem:
        try:
            job_id_str = filename_stem.rsplit("_", 1)[-1]
            job_id = int(job_id_str)
            result["job_id"] = job_id
            logger.debug(f"Extracted job_id {job_id} from {log_file.name}")
            
            # Match with jobs_ci_metadata to get job_conclusion
            if jobs_ci_metadata:
                for job_info in jobs_ci_metadata:
                    if job_info.get("job_id") == job_id:
                        result["job_conclusion"] = job_info.get("job_conclusion")
                        logger.debug(f"Matched job_id {job_id} with conclusion: {result['job_conclusion']}")
                        break
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not extract job_id from filename {log_file.name}: {e}")
    
    try:
        logger.debug(f"Parsing CI job log: {log_file.name}")
        content = log_file.read_text(errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read log file {log_file.name}: {e}")
        return result
    
    content = _strip_ansi(content)
    lines = content.splitlines()
    
    # Extract runner name (e.g., "Runner name: 'e08cs05'")
    for line in lines:
        stripped = _strip_timestamp_prefix(line).strip()
        runner_match = re.match(r"Runner name:\s*['\"]?([^'\"]+)['\"]?", stripped)
        if runner_match:
            result["runner_name"] = runner_match.group(1).strip()
            logger.debug(f"Found runner name: {result['runner_name']}")
            break
    
    # Extract docker image from Inputs section
    in_inputs_section = False
    for i, line in enumerate(lines):
        stripped = _strip_timestamp_prefix(line).strip()
        
        # Detect start of Inputs section
        if "##[group] Inputs" in line or "##[group]Inputs" in line:
            in_inputs_section = True
            continue
        
        # Detect end of Inputs section
        if in_inputs_section and "##[endgroup]" in line:
            in_inputs_section = False
            continue
        
        # Look for 'image:' line within Inputs section
        if in_inputs_section:
            image_match = re.match(r'\s*image:\s*(.+)', stripped)
            if image_match:
                result["docker_image"] = image_match.group(1).strip()
                logger.debug(f"Found docker image: {result['docker_image']}")
                break
    
    # Extract tt-smi output
    for i, line in enumerate(lines):
        if "tt-smi-metal -s" in line:
            # Look for the JSON block starting with '{'
            json_lines = []
            found_start = False
            brace_count = 0
            
            for j in range(i + 1, len(lines)):
                stripped_line = _strip_timestamp_prefix(lines[j])
                
                if not found_start:
                    if stripped_line.strip() == "{":
                        found_start = True
                        json_lines.append(stripped_line)
                        brace_count = 1
                else:
                    json_lines.append(stripped_line)
                    brace_count += stripped_line.count("{")
                    brace_count -= stripped_line.count("}")
                    
                    if brace_count == 0:
                        break
            
            if json_lines and found_start and brace_count == 0:
                try:
                    json_text = "\n".join(json_lines)
                    tt_smi_output = json.loads(json_text)
                    
                    # Validate that we have a proper tt-smi output structure
                    if isinstance(tt_smi_output, dict):
                        # Extract firmware_bundle from device_info
                        device_info = tt_smi_output.get("device_info", [])
                        if device_info and isinstance(device_info, list) and len(device_info) > 0:
                            first_device = device_info[0]
                            if isinstance(first_device, dict):
                                firmwares = first_device.get("firmwares", {})
                                if isinstance(firmwares, dict):
                                    result["firmware_bundle"] = firmwares.get("fw_bundle_version")
                        
                        # Extract kmd_version from host_info.Driver
                        host_info = tt_smi_output.get("host_info", {})
                        if isinstance(host_info, dict):
                            driver_str = host_info.get("Driver")
                            if driver_str and isinstance(driver_str, str):
                                result["kmd_version"] = driver_str
                        
                        # Remove device_info before storing
                        tt_smi_output.pop("device_info", None)
                        result["tt_smi_output"] = tt_smi_output
                        
                        logger.debug(f"Found tt-smi output in {log_file.name}")
                        logger.debug(f"  firmware_bundle: {result['firmware_bundle']}")
                        logger.debug(f"  kmd_version: {result['kmd_version']}")
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse tt-smi JSON from {log_file.name}: {e}")
            break
    
    return result


def parse_ci_job_log(logs_dir: Path, jobs_ci_metadata: Optional[List[dict]] = None) -> dict:
    """Parse all CI job logs from a directory and aggregate system information.
    
    Args:
        logs_dir: Directory containing job log .txt files
        jobs_ci_metadata: Optional list of job metadata dicts from jobs_ci_metadata.json
        
    Returns:
        Dict with aggregated CI logs data containing:
            - docker_image: Docker image used
            - build_runner_name: Runner name from build job
            - runner_names_by_job_id: Dict mapping job_id to runner_name
            - firmware_bundle: Firmware bundle version
            - kmd_version: KMD version string
            - ci_log_tt_smi_output: tt-smi output dict
    """
    ci_logs_dict = {
        "docker_image": None,
        "build_runner_name": None,
        "runner_names_by_job_id": {},  # Map job_id -> runner_name
        "firmware_bundle": None,
        "kmd_version": None,
        "ci_log_tt_smi_output": None,
    }
    
    log_files = sorted(logs_dir.glob("*.txt"))
    logger.info(f"Parsing {len(log_files)} log files from: {logs_dir}")
    
    for log_file in log_files:
        job_log_data = _parse_single_ci_job_log(log_file, jobs_ci_metadata)
        
        # Map job_id to runner_name for easy lookup
        if job_log_data["job_id"] is not None and job_log_data["runner_name"]:
            ci_logs_dict["runner_names_by_job_id"][job_log_data["job_id"]] = job_log_data["runner_name"]
        
        # Use first non-None value found for each field
        if not ci_logs_dict["docker_image"] and job_log_data["docker_image"]:
            ci_logs_dict["docker_image"] = job_log_data["docker_image"]
            logger.info(f"Found docker image from {log_file.name}: {ci_logs_dict['docker_image']}")
        
        if not ci_logs_dict["ci_log_tt_smi_output"] and job_log_data["tt_smi_output"]:
            ci_logs_dict["ci_log_tt_smi_output"] = job_log_data["tt_smi_output"]
            logger.info(f"Found tt-smi output from {log_file.name}")
        
        if not ci_logs_dict["firmware_bundle"] and job_log_data["firmware_bundle"]:
            ci_logs_dict["firmware_bundle"] = job_log_data["firmware_bundle"]
            logger.info(f"Found firmware bundle from {log_file.name}: {ci_logs_dict['firmware_bundle']}")
        
        if not ci_logs_dict["kmd_version"] and job_log_data["kmd_version"]:
            ci_logs_dict["kmd_version"] = job_log_data["kmd_version"]
            logger.info(f"Found KMD version from {log_file.name}: {ci_logs_dict['kmd_version']}")
        
        # Extract build runner name from build-inference-server jobs
        if not ci_logs_dict["build_runner_name"] and "build-inference-server" in log_file.name.lower():
            if job_log_data["runner_name"]:
                ci_logs_dict["build_runner_name"] = job_log_data["runner_name"]
                logger.info(f"Found build runner name: {ci_logs_dict['build_runner_name']}")
    
    return ci_logs_dict


def format_dt(dt_str: str) -> str:
    # Convert ISO to YYYY-MM-DD_HH-MM-SS
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except Exception:
        return dt_str.replace(":", "-").replace("T", "_").replace("Z", "")


def parse_job_name(job_name: str) -> Optional[dict]:
    """Parse job name to extract workflow details.
    
    Job name format: "run-{workflow}-{hardware_name} / test ({workflow_type}, {model_name}, {hardware_name}, {hardware})"
    
    Args:
        job_name: GitHub Actions job name string
        
    Returns:
        Dict with workflow_type, model_name, hardware_name, hardware or None if parse fails
    """
    # Pattern to match: "prefix / test (workflow_type, model_name, hardware_name, hardware)"
    pattern = r'.+\s*/\s*test\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)'
    match = re.match(pattern, job_name)
    
    if not match:
        logger.debug(f"Could not parse job name: {job_name}")
        return None
    
    workflow_type = match.group(1).strip()
    model_name = match.group(2).strip()
    hardware_name = match.group(3).strip()
    hardware = match.group(4).strip()
    
    return {
        "workflow_type": workflow_type,
        "model_name": model_name,
        "hardware_name": hardware_name,
        "hardware": hardware,
    }


def match_jobs_to_workflow_logs(jobs_ci_metadata: List[dict], workflow_logs_dir_name: str) -> Optional[dict]:
    """Match a job from jobs_ci_metadata to a workflow_logs directory.
    
    Workflow logs dir format: workflow_logs_{workflow_type}_{model_name}_{hardware_name}
    
    Args:
        jobs_ci_metadata: List of job metadata dicts from jobs_ci_metadata.json
        workflow_logs_dir_name: Name of workflow_logs directory
        
    Returns:
        Matched job dict with added fields: model_name, hardware, hardware_name or None
    """
    if not jobs_ci_metadata:
        logger.debug(f"No jobs_ci_metadata provided for matching")
        return None
    
    # Parse workflow_logs_dir_name: workflow_logs_{workflow_type}_{model_name}_{hardware_name}
    if not workflow_logs_dir_name.startswith("workflow_logs_"):
        logger.warning(f"Invalid workflow_logs dir name format: {workflow_logs_dir_name}")
        return None
    
    # Remove "workflow_logs_" prefix
    remainder = workflow_logs_dir_name[len("workflow_logs_"):]
    
    # Split by underscore to extract: workflow_type, model_name, hardware_name
    # Format is: {workflow_type}_{model_name}_{hardware_name}
    # But model_name can contain underscores, so we need to be careful
    # We know workflow_type is usually "release", "benchmarks", etc.
    # hardware_name is the last component (llmbox, n300, n150, t3k, tg, etc.)
    
    parts = remainder.rsplit("_", 1)
    if len(parts) != 2:
        logger.warning(f"Could not parse workflow_logs dir name: {workflow_logs_dir_name}")
        return None
    
    model_and_workflow = parts[0]
    hardware_name = parts[1]
    
    # Extract workflow_type (first component before model name)
    workflow_parts = model_and_workflow.split("_", 1)
    if len(workflow_parts) != 2:
        logger.warning(f"Could not extract workflow_type from: {model_and_workflow}")
        return None
    
    workflow_type = workflow_parts[0]
    model_name = workflow_parts[1].replace("_", "-")  # Convert underscores back to hyphens
    
    logger.debug(f"Parsed workflow_logs dir: workflow_type={workflow_type}, model_name={model_name}, hardware_name={hardware_name}")
    
    # Match against jobs_ci_metadata
    for job in jobs_ci_metadata:
        job_name = job.get("job_name", "")
        parsed_job = parse_job_name(job_name)
        
        if not parsed_job:
            continue
        
        # Match when workflow_type, model_name, and hardware_name all match
        if (parsed_job["workflow_type"] == workflow_type and
            parsed_job["model_name"] == model_name and
            parsed_job["hardware_name"] == hardware_name):
            
            # Create matched job info with parsed fields
            matched_job = {
                "job_id": job.get("job_id"),
                "job_name": job.get("job_name"),
                "job_status": job.get("job_status"),
                "job_conclusion": job.get("job_conclusion"),
                "job_url": job.get("job_url"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "model_name": parsed_job["model_name"],
                "hardware": parsed_job["hardware"],
                "hardware_name": parsed_job["hardware_name"],
            }
            
            logger.info(f"Matched job {job.get('job_id')} to {workflow_logs_dir_name}")
            return matched_job
    
    logger.warning(f"Could not match any job to {workflow_logs_dir_name}")
    return None


def check_auth(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> str:
    """
    Verify GitHub token authentication and access to the target repository.

    This function must be called before any GitHub API operations to ensure
    the token has proper permissions and access to the repository.

    Args:
        owner: GitHub repository owner/organization
        repo: GitHub repository name

    Returns:
        str: The validated token

    Raises:
        SystemExit: If authentication fails or token is not set
    """
    logger = logging.getLogger(__name__)

    # Step 1: Check if token environment variable is set
    token = os.getenv("GH_PAT")
    if not token:
        logger.error("❌ GITHUB TOKEN NOT FOUND")
        logger.error("=" * 50)
        logger.error("The GH_PAT environment variable is not set.")
        logger.error("")
        logger.error("🔧 REQUIRED SETUP:")
        logger.error("   1. Create a GitHub Personal Access Token:")
        logger.error("      - Go to: https://github.com/settings/tokens")
        logger.error("      - Click 'Generate new token'")
        logger.error("      - Select 'Fine-grained tokens' (recommended) or 'Classic'")
        logger.error("")
        logger.error("   2. Configure token permissions:")
        logger.error("      For FINE-GRAINED tokens:")
        logger.error("      - Repository access: Select specific repository or 'All repositories'")
        logger.error("      - Permissions needed: Actions: Read, Contents: Read, Metadata: Read")
        logger.error("")
        logger.error("      For CLASSIC tokens:")
        logger.error("      - Select scopes: 'repo' (for private repositories)")
        logger.error("")
        logger.error("   3. Set environment variable:")
        logger.error("      export GH_PAT='your_github_token_here'")
        logger.error("      # OR set it in your shell profile (~/.bashrc, ~/.zshrc)")
        logger.error("")
        logger.error("   4. Verify token works:")
        logger.error("      curl -H 'Authorization: Bearer $GH_PAT' \\")
        logger.error("           -H 'Accept: application/vnd.github+json' \\")
        logger.error("           https://api.github.com/user")
        logger.error("")
        logger.error("💡 TROUBLESHOOTING:")
        logger.error("   - Ensure token is from the same GitHub account with repository access")
        logger.error("   - Check token hasn't expired")
        logger.error("   - Verify you can access the repository in a web browser")
        logger.error("   - For organization repos, ensure your account is a member")
        logger.error("=" * 50)
        raise SystemExit(1)

    # Step 2: Test token validity and repository access
    logger.info(f"   Found GH_PAT token (length: {len(token)} characters)")
    logger.info(f"   Testing access to repository: {owner}/{repo}")

    test_url = f"{GITHUB_API}/repos/{owner}/{repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "models-ci-reader/1.0",
        "Accept": "application/vnd.github+json"
    }

    try:
        req = Request(test_url, headers=headers, method="GET")
        with urlopen(req, timeout=30) as resp:
            if resp.getcode() == 200:
                logger.info("✅ Repository access test: SUCCESS")
                # Get basic repo info for verification
                data = json.loads(resp.read().decode("utf-8"))
                repo_name = data.get("full_name", "unknown")
                is_private = data.get("private", False)
                logger.info(f"   Repository: {repo_name}")
                logger.info(f"   Private: {is_private}")
                logger.info(f"   Description: {data.get('description', 'None')}")
            else:
                logger.error(f"❌ Repository access test: FAILED (HTTP {resp.getcode()})")
                _handle_auth_error(resp.getcode(), owner, repo, token)
                raise SystemExit(1)

    except HTTPError as e:
        logger.error(f"❌ Repository access test: FAILED")
        _handle_auth_error(e.code, owner, repo, token)
        raise SystemExit(1)

    except Exception as e:
        logger.error(f"❌ Unexpected error during authentication test: {e}")
        logger.error("   This might be a network connectivity issue")
        raise SystemExit(1)

    # Step 3: Test Actions and Workflow access (this is what we actually need)
    logger.info("   Testing GitHub Actions and Workflow access...")

    # Test 1: Check Actions permission (basic Actions API access)
    actions_url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows"
    has_workflows = False

    try:
        req = Request(actions_url, headers=headers, method="GET")
        with urlopen(req, timeout=30) as resp:
            if resp.getcode() == 200:
                logger.info("✅ Actions API access: SUCCESS")
                data = json.loads(resp.read().decode("utf-8"))
                workflows = data.get("workflows", [])
                workflow_count = len(workflows)
                logger.info(f"   Found {workflow_count} workflows")

                if workflow_count > 0:
                    has_workflows = True
                    logger.info("   Available workflows:")
                    for wf in workflows[:3]:  # Show first 3 workflows
                        logger.info(f"     - {wf.get('name', 'Unnamed')} (ID: {wf.get('id')})")
                    if workflow_count > 3:
                        logger.info(f"     ... and {workflow_count - 3} more")
                else:
                    logger.warning("⚠️  Repository has no workflows configured")

            else:
                logger.error(f"❌ Actions API access: FAILED (HTTP {resp.getcode()})")
                logger.error("   Cannot access workflows - this will prevent script execution")
                _handle_workflow_access_error(resp.getcode(), owner, repo, token)
                raise SystemExit(1)

    except HTTPError as e:
        if e.code == 404:
            logger.error("❌ Actions API access: FAILED")
            logger.error("   Repository either has no workflows or you lack Actions permissions")
            logger.error("   This script requires Actions access to function properly")
            _handle_workflow_access_error(404, owner, repo, token)
            raise SystemExit(1)
        else:
            logger.error(f"❌ Actions API access: FAILED (HTTP {e.code})")
            _handle_workflow_access_error(e.code, owner, repo, token)
            raise SystemExit(1)

    # Test 2: If workflows exist, test workflow runs access
    if has_workflows:
        try:
            # Get the first workflow to test runs access
            first_workflow = workflows[0]
            workflow_id = first_workflow.get("id")
            runs_url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs?per_page=1"

            req = Request(runs_url, headers=headers, method="GET")
            with urlopen(req, timeout=30) as resp:
                if resp.getcode() == 200:
                    logger.info("✅ Workflow runs access: SUCCESS")
                    runs_data = json.loads(resp.read().decode("utf-8"))
                    runs_count = len(runs_data.get("workflow_runs", []))
                    logger.info(f"   Successfully accessed workflow runs (found {runs_count} recent runs)")
                else:
                    logger.warning(f"⚠️  Workflow runs access: HTTP {resp.getcode()}")
                    logger.warning("   Can list workflows but cannot access run data")

        except HTTPError as e:
            if e.code == 404:
                logger.info("ℹ️  Workflow has no runs or cannot access run data")
            else:
                logger.warning(f"⚠️  Workflow runs access: HTTP {e.code} - {e.reason}")

    # Test 3: Test artifacts access (important for downloading workflow artifacts)
    if has_workflows:
        try:
            # Try to list runs first to get a recent run ID for artifact testing
            runs_url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs?per_page=5"
            req = Request(runs_url, headers=headers, method="GET")
            with urlopen(req, timeout=30) as resp:
                if resp.getcode() == 200:
                    runs_data = json.loads(resp.read().decode("utf-8"))
                    recent_runs = runs_data.get("workflow_runs", [])

                    if recent_runs:
                        # Test artifacts access on the most recent run
                        test_run_id = recent_runs[0].get("id")
                        artifacts_url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{test_run_id}/artifacts"

                        req = Request(artifacts_url, headers=headers, method="GET")
                        with urlopen(req, timeout=30) as resp:
                            if resp.getcode() == 200:
                                logger.info("✅ Workflow artifacts access: SUCCESS")
                                artifacts_data = json.loads(resp.read().decode("utf-8"))
                                artifacts_count = len(artifacts_data.get("artifacts", []))
                                logger.info(f"   Successfully accessed run artifacts (found {artifacts_count} artifacts)")
                            else:
                                logger.warning(f"⚠️  Workflow artifacts access: HTTP {resp.getcode()}")
                    else:
                        logger.info("ℹ️  No recent runs found to test artifact access")
                else:
                    logger.warning(f"⚠️  Cannot access recent runs: HTTP {resp.getcode()}")

        except HTTPError as e:
            if e.code == 404:
                logger.info("ℹ️  No artifacts found or cannot access artifact data")
            else:
                logger.warning(f"⚠️  Artifacts access test: HTTP {e.code} - {e.reason}")

    logger.info("✅ GitHub Actions and Workflow access verification: COMPLETE")
    return token


def _handle_workflow_access_error(error_code: int, owner: str, repo: str, token: str):
    """Helper function to provide detailed error messages for workflow access failures"""
    logger = logging.getLogger(__name__)
    logger.error("")
    logger.error("🔧 WORKFLOW ACCESS TROUBLESHOOTING:")
    logger.error("   This script requires GitHub Actions and workflow access to function.")
    logger.error("   Without these permissions, the script cannot:")
    logger.error("   - List workflows in the repository")
    logger.error("   - Access workflow run data")
    logger.error("   - Download workflow artifacts and logs")
    logger.error("")

    if error_code == 401:
        logger.error("   🔑 ACTIONS PERMISSIONS ERROR:")
        logger.error("   Your token lacks GitHub Actions permissions")
        logger.error("")
        logger.error("   🔧 REQUIRED PERMISSIONS:")
        logger.error("   For FINE-GRAINED tokens:")
        logger.error("   - Repository access: Select the target repository")
        logger.error("   - Actions: READ permission")
        logger.error("   - Contents: READ permission (to access workflow files)")
        logger.error("   - Metadata: READ permission")
        logger.error("")
        logger.error("   For CLASSIC tokens:")
        logger.error("   - Select 'repo' scope (provides Actions access)")
        logger.error("")
        logger.error("   💡 QUICK FIX:")
        logger.error("   1. Go to https://github.com/settings/tokens?type=beta")
        logger.error("   2. Create new fine-grained token with:")
        logger.error("      Repository: tenstorrent/tt-shield")
        logger.error("      Actions: ✓ Read")
        logger.error("      Contents: ✓ Read")
        logger.error("      Metadata: ✓ Read")
        logger.error("   3. Update GH_PAT with new token")

    elif error_code == 404:
        logger.error("   📁 REPOSITORY ACCESS ERROR:")
        logger.error("   Cannot access workflows in this repository")
        logger.error("")
        logger.error("   🔧 POSSIBLE CAUSES:")
        logger.error("   1. Repository has no workflows configured")
        logger.error("   2. Repository is private and you lack access")
        logger.error("   3. GitHub Actions are disabled for this repository")
        logger.error("   4. Repository name or owner is incorrect")
        logger.error("")
        logger.error("   💡 CHECKLIST:")
        logger.error("   - Verify repository URL: https://github.com/" + f"{owner}/{repo}")
        logger.error("   - Check if Actions are enabled in repository settings")
        logger.error("   - Ensure you have read access to the repository")

    elif error_code == 403:
        logger.error("   🚫 FORBIDDEN ACCESS:")
        logger.error("   GitHub Actions access is forbidden")
        logger.error("")
        logger.error("   🔧 LIKELY CAUSES:")
        logger.error("   1. Organization requires specific Actions permissions")
        logger.error("   2. Repository has restricted Actions access")
        logger.error("   3. Token lacks required scopes for Actions")
        logger.error("   4. Organization membership required for Actions access")

    else:
        logger.error(f"   ❓ UNKNOWN ERROR (HTTP {error_code}):")
        logger.error("   This is an unexpected error code for workflow access")

    logger.error("")
    logger.error("   🧪 TEST YOUR TOKEN:")
    logger.error("   curl -H 'Authorization: Bearer $GH_PAT' \\")
    logger.error("        -H 'Accept: application/vnd.github+json' \\")
    logger.error("        https://api.github.com/repos/" + f"{owner}/{repo}" + "/actions/workflows")
    logger.error("")
    logger.error("   🔍 QUICK DIAGNOSIS:")
    logger.error("   1. Check if you can access the repo: https://github.com/" + f"{owner}/{repo}")
    logger.error("   2. Verify Actions are enabled in repository settings")
    logger.error("   3. Ensure your token has the right permissions for Actions")


def _handle_auth_error(error_code: int, owner: str, repo: str, token: str):
    """Helper function to provide detailed error messages for authentication failures"""
    logger = logging.getLogger(__name__)

    if error_code == 401:
        logger.error("   🔑 AUTHENTICATION ERROR:")
        logger.error(f"   Your GitHub token cannot access {owner}/{repo}")
        logger.error("")
        logger.error("   🔧 SOLUTIONS:")
        logger.error("   1. Verify token has correct permissions:")
        logger.error("      - Fine-grained: Repository access + Actions/Content/Metadata read")
        logger.error("      - Classic: 'repo' scope for private repositories")
        logger.error("")
        logger.error("   2. Check token expiration:")
        logger.error("      - Go to: https://github.com/settings/tokens")
        logger.error("      - Regenerate token if expired")
        logger.error("")
        logger.error("   3. Verify repository access:")
        logger.error("      - Ensure you can view the repo at: https://github.com/" + f"{owner}/{repo}")
        logger.error("      - Check if you're in the organization (if applicable)")
        logger.error("")
        logger.error("   4. Test token manually:")
        logger.error("      curl -H 'Authorization: Bearer $GH_PAT' \\")
        logger.error("           https://api.github.com/user")
        logger.error("")
        logger.error("   5. Token length check:")
        logger.error(f"      Current token length: {len(token)} characters")
        logger.error("      Token preview: " + f"{token[:10]}...{token[-4:]}")

    elif error_code == 404:
        logger.error("   📁 REPOSITORY NOT FOUND:")
        logger.error(f"   Cannot find repository: {owner}/{repo}")
        logger.error("")
        logger.error("   🔧 SOLUTIONS:")
        logger.error("   1. Verify repository name and owner are correct")
        logger.error("   2. Check if repository is private and you have access")
        logger.error("   3. Ensure the repository hasn't been deleted or moved")
        logger.error("   4. Try accessing the repo URL in your browser:")
        logger.error("      https://github.com/" + f"{owner}/{repo}")

    elif error_code == 403:
        logger.error("   🚫 FORBIDDEN ACCESS:")
        logger.error(f"   Access denied to {owner}/{repo}")
        logger.error("   This usually means insufficient permissions")
        logger.error("")
        logger.error("   🔧 SOLUTIONS:")
        logger.error("   1. Check if you need organization membership")
        logger.error("   2. Verify token has 'repo' scope (classic) or repository access (fine-grained)")
        logger.error("   3. Contact repository administrators for access")

    else:
        logger.error(f"   ❓ UNKNOWN ERROR (HTTP {error_code}):")
        logger.error("   This is an unexpected error code")
        logger.error("   Check GitHub API status: https://www.githubstatus.com/")


def process_run_directory(run_out_dir: Path, run_ts_str: str, run_ci_metadata: Optional[dict] = None) -> Dict[str, List[dict]]:
    """Process a single run directory and return all models data.
    
    Args:
        run_out_dir: Path to the run directory
        run_ts_str: Timestamp string for the run
        run_ci_metadata: Optional metadata dict with ci_run_id, owner, repo
    
    Returns:
        Dict mapping model_id to list of model entries
    """
    all_models_dict: Dict[str, List[dict]] = {}
    
    # Load jobs metadata if available
    jobs_ci_metadata_path = run_out_dir / "jobs_ci_metadata.json"
    jobs_ci_metadata: Optional[List[dict]] = None
    if jobs_ci_metadata_path.exists():
        try:
            logger.info(f"Loading jobs metadata from: {jobs_ci_metadata_path}")
            jobs_ci_metadata = json.loads(jobs_ci_metadata_path.read_text())
            logger.info(f"Loaded metadata for {len(jobs_ci_metadata)} jobs")
        except Exception as e:
            logger.warning(f"Failed to load jobs metadata: {e}")
    
    # Parse ci logs if they exist
    ci_logs_dir = run_out_dir / "logs"
    
    if ci_logs_dir.exists():
        ci_logs_dict = parse_ci_job_log(ci_logs_dir, jobs_ci_metadata)
    else:
        logger.warning(f"No logs directory found at: {ci_logs_dir}")
        ci_logs_dict = {
            "docker_image": None,
            "build_runner_name": None,
            "runner_names_by_job_id": {},
            "firmware_bundle": None,
            "kmd_version": None,
            "ci_log_tt_smi_output": None,
        }

    
    
    # Process all workflow_logs_* artifact directories
    artifact_dirs = [d for d in run_out_dir.iterdir() if d.is_dir() and d.name.startswith("workflow_logs_")]
    logger.info(f"Found {len(artifact_dirs)} artifact directories in {run_out_dir.name}")
    
    for art_dir in artifact_dirs:
        logger.info(f"Processing artifact directory: {art_dir.name}")
        
        # Use new workflow_logs_parser module
        workflow_logs_dict = parse_workflow_logs_dir(art_dir)
        if not workflow_logs_dict:
            logger.warning(f"Failed to parse {art_dir.name}, skipping")
            continue
        
        model_id = workflow_logs_dict["summary"]["model_id"]
        
        # Match job from jobs_ci_metadata to this workflow_logs directory
        matched_job_info = None
        if jobs_ci_metadata:
            dir_name = workflow_logs_dict.get("dir_name")
            if dir_name:
                matched_job_info = match_jobs_to_workflow_logs(jobs_ci_metadata, dir_name)
        
        # Create model-specific ci_metadata with job metadata
        model_ci_metadata = None
        if run_ci_metadata:
            ci_run_id = run_ci_metadata.get("run_id")
            owner = run_ci_metadata.get("owner")
            repo = run_ci_metadata.get("repo")
            if ci_run_id and owner and repo:
                ci_run_link = f"https://github.com/{owner}/{repo}/actions/runs/{ci_run_id}"
                logger.info(f"   Using run metadata: run_id={ci_run_id}, link={ci_run_link}")
            else:
                logger.warning(f"   Incomplete run metadata: run_id={ci_run_id}, owner={owner}, repo={repo}")
                ci_run_link = None
            
            # Create model-specific metadata without the full jobs list
            model_ci_metadata = {
                "run_id": run_ci_metadata.get("run_id"),
                "run_number": run_ci_metadata.get("run_number"),
                "owner": run_ci_metadata.get("owner"),
                "repo": run_ci_metadata.get("repo"),
                "workflow_file": run_ci_metadata.get("workflow_file"),
                "created_at": run_ci_metadata.get("created_at"),
                "updated_at": run_ci_metadata.get("updated_at"),
                "ci_run_link": ci_run_link,
                "ci_job_metadata": matched_job_info,
            }
        else:
            logger.debug(f"   No run metadata available - ci_run_id and ci_run_link will be null")
        
        # Create model-specific ci_logs with only the relevant runner_name
        model_ci_logs = {
            "docker_image": ci_logs_dict.get("docker_image"),
            "build_runner_name": ci_logs_dict.get("build_runner_name"),
            "firmware_bundle": ci_logs_dict.get("firmware_bundle"),
            "kmd_version": ci_logs_dict.get("kmd_version"),
            "ci_log_tt_smi_output": ci_logs_dict.get("ci_log_tt_smi_output"),
        }
        
        # Extract runner_name for this specific job using job_id
        runner_name = None
        if matched_job_info:
            job_id = matched_job_info.get("job_id")
            if job_id and ci_logs_dict.get("runner_names_by_job_id"):
                runner_name = ci_logs_dict["runner_names_by_job_id"].get(job_id)
        model_ci_logs["runner_name"] = runner_name
        
        # Store ALL models (not just passing ones)
        entry = {
            "job_run_datetimestamp": run_ts_str,
            "ci_metadata": model_ci_metadata,
            "ci_logs": model_ci_logs,
            "workflow_logs": workflow_logs_dict,
        }
        all_models_dict.setdefault(model_id, []).append(entry)
    
    return all_models_dict


def download_runs(owner: str, repo: str, workflow_file: str, token: str, out_root: Path, max_runs: int, run_id: Optional[int]) -> None:
    """Download workflow runs without processing them."""
    # Resolve workflow and list recent runs
    wf = get_workflow(owner, repo, workflow_file, token)
    workflow_id = wf.get("id")
    assert workflow_id, "Could not resolve workflow id"
    
    runs: List[dict]
    if run_id is not None:
        logger.info(f"Fetching single workflow run by ID: {run_id}")
        run_obj = get_workflow_run(run_id, owner, repo, token)
        run_workflow_file = (run_obj.get("name") or "").lower()
        logger.info(f"Fetched run {run_id} (workflow name: '{run_workflow_file}')")
        runs = [run_obj]
    else:
        runs = list_workflow_runs(workflow_id, owner, repo, token, per_page=max_runs)
    
    for run in runs:
        run_id = run.get("id")
        run_number = run.get("run_number")
        
        # Create run dir e.g. On_nightly_236
        run_dir_name = f"On_nightly_{run_number}"
        run_out_dir = out_root / run_dir_name
        ensure_dir(run_out_dir)
        logger.info(f"Run output directory: {run_out_dir}")
        
        # Save run metadata for later processing
        run_ci_metadata = {
            "run_id": run_id,
            "run_number": run_number,
            "owner": owner,
            "repo": repo,
            "workflow_file": workflow_file,
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
        }
        metadata_path = run_out_dir / "run_ci_metadata.json"
        logger.info(f"Saving run metadata to: {metadata_path}")
        metadata_path.write_text(json.dumps(run_ci_metadata, indent=2))
        
        # Download consolidated run logs and save job metadata
        logs_dir = run_out_dir / "logs"
        try:
            if logs_dir.exists():
                logger.info(f"Removing existing logs directory: {logs_dir}")
                shutil.rmtree(logs_dir)
            ensure_dir(logs_dir)
            
            # Get all jobs and save their metadata
            logger.info(f"Listing all jobs for run {run_id}")
            jobs = list_run_jobs(run_id, owner, repo, token)
            logger.info(f"Found {len(jobs)} total jobs")
            
            # Extract job metadata for saving
            jobs_ci_metadata = []
            for job in jobs:
                job_info = {
                    "job_id": job.get("id"),
                    "job_name": job.get("name"),
                    "job_status": job.get("status"),
                    "job_conclusion": job.get("conclusion"),
                    "job_url": job.get("html_url"),
                    "started_at": job.get("started_at"),
                    "completed_at": job.get("completed_at"),
                }
                jobs_ci_metadata.append(job_info)
            
            # Save jobs metadata to file
            jobs_ci_metadata_path = run_out_dir / "jobs_ci_metadata.json"
            logger.info(f"Saving jobs metadata to: {jobs_ci_metadata_path}")
            jobs_ci_metadata_path.write_text(json.dumps(jobs_ci_metadata, indent=2))
            logger.info(f"Saved metadata for {len(jobs_ci_metadata)} jobs")
            
            # Download individual job logs
            download_ci_job_logs(jobs, logs_dir, owner, repo, token)
            
        except Exception as e:
            logger.warning(f"Failed to process run logs: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # List and download artifacts matching workflow logs
        try:
            artifacts = list_run_artifacts(run_id, owner, repo, token)
            logger.info(f"Found {len(artifacts)} artifacts for run {run_id}")
        except Exception:
            artifacts = []
        
        for artifact in artifacts:
            name = artifact.get("name", "")
            if not name.startswith("workflow_logs_"):
                continue
            artifact_id = artifact.get("id")
            archive_url = artifact.get("archive_download_url")
            if not artifact_id and not archive_url:
                continue
            try:
                ref = archive_url if archive_url else artifact_id
                z = download_artifact_zip(ref, owner, repo, token)
                art_dir = run_out_dir / name
                if art_dir.exists():
                    shutil.rmtree(art_dir)
                extract_zip_to_dir(z, art_dir)
                logger.info(f"  ✓ {name}")
            except Exception as e:
                logger.warning(f"  ✗ {name}: {e}")
                continue


def process_all_runs(out_root: Path, owner: str, repo: str, last_run_only: bool = False) -> Tuple[Dict[str, List[dict]], List[str]]:
    """Process all run directories in out_root.
    
    Args:
        out_root: Root directory containing On_nightly_* run directories
        owner: GitHub repository owner
        repo: GitHub repository name
        last_run_only: If True, only process the most recent run directory
        
    Returns:
        Tuple of (all_models_dict, all_run_timestamps)
    """
    logger.info(f"Processing run directories in: {out_root} (last_run_only={last_run_only})")
    
    all_models_dict: Dict[str, List[dict]] = {}
    all_run_timestamps: List[str] = []
    
    # Find all On_nightly_* directories
    run_dirs = sorted([d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("On_nightly_")])
    logger.info(f"Found {len(run_dirs)} run directories")
    
    # If last_run_only mode, filter to only the most recent run
    if last_run_only and run_dirs:
        # Sort by modification time (most recent last) and take the last one
        run_dirs = sorted(run_dirs, key=lambda d: d.stat().st_mtime)
        run_dirs = [run_dirs[-1]]
        logger.info(f"Last run only mode: processing only {run_dirs[0].name}")
    
    logger.info(f"Processing {len(run_dirs)} run directories")
    
    for run_out_dir in run_dirs:
        logger.info(f"Processing run directory: {run_out_dir.name}")
        
        # Load run metadata if available
        run_ci_metadata: Optional[dict] = None
        metadata_path = run_out_dir / "run_ci_metadata.json"
        if metadata_path.exists():
            try:
                logger.info(f"Loading run metadata from: {metadata_path}")
                run_ci_metadata = json.loads(metadata_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load run metadata: {e}")
        else:
            logger.warning(f"⚠️  No run_ci_metadata.json found for {run_out_dir.name}")
            logger.warning(f"   This may be from an older download before metadata saving was implemented.")
            logger.warning(f"   ci_run_id and ci_run_link will be null for this run.")
            logger.warning(f"   To fix: re-download artifacts for this run using the script without --process flag.")
        
        # Extract timestamp from directory metadata or use current time
        run_ts_str = datetime.fromtimestamp(run_out_dir.stat().st_mtime).strftime("%Y-%m-%d_%H-%M-%S")
        all_run_timestamps.append(run_ts_str)
        
        # Process this run directory
        run_all_models_dict = process_run_directory(run_out_dir, run_ts_str, run_ci_metadata)
        
        # Merge results
        for model_id, entries in run_all_models_dict.items():
            all_models_dict.setdefault(model_id, []).extend(entries)
    
    return all_models_dict, all_run_timestamps


def write_summary_output(all_models_dict: Dict[str, List[dict]], all_run_timestamps: List[str], out_root: Path) -> None:
    """Write summary JSON files for all models and last good passing models."""
    if all_run_timestamps:
        earliest = min(all_run_timestamps)
        latest = max(all_run_timestamps)
    else:
        now_s = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        earliest = latest = now_s
    
    # Write full output with ALL models (passing and non-passing)
    all_models_name = f"models_ci_all_results_{earliest}_to_{latest}.json"
    all_models_path = out_root / all_models_name
    logger.info(f"Writing all models JSON to: {all_models_path}")
    all_models_text = json.dumps(all_models_dict, indent=2)
    all_models_path.write_text(all_models_text)
    logger.info(f"Wrote {len(all_models_text.encode('utf-8'))} bytes to {all_models_path}")
    
    # Create last good passing models summary (for backward compatibility)
    models_ci_last_good: Dict[str, dict] = {}
    for model_id, entries in all_models_dict.items():
        # Filter to only passing entries
        passing_entries = [e for e in entries if e.get("workflow_logs", {}).get("summary", {}).get("is_passing", False)]
        if not passing_entries:
            continue
        
        # Choose entry with max job_run_datetimestamp
        entries_sorted = sorted(passing_entries, key=lambda e: e.get("job_run_datetimestamp", ""))
        chosen = entries_sorted[-1]
        workflow_data = chosen.get("workflow_logs", {}).get("summary", {})
        ci_metadata = chosen.get("ci_metadata") or {}
        ci_logs = chosen.get("ci_logs") or {}
        models_ci_last_good[model_id] = {
            "tt_metal_commit": workflow_data.get("tt_metal_commit"),
            "vllm_commit": workflow_data.get("vllm_commit"),
            "docker_image": workflow_data.get("docker_image"),
            "ci_run_id": ci_metadata.get("run_id") if ci_metadata else None,
            "ci_run_link": ci_metadata.get("ci_run_link") if ci_metadata else None,
            "perf_status": workflow_data.get("perf_status"),
            "accuracy_status": workflow_data.get("accuracy_status"),
            "minimum_firmware_bundle": ci_logs.get("firmware_bundle") if ci_logs else None,
            "minimum_driver_version": ci_logs.get("kmd_version") if ci_logs else None,
        }
    
    # Write models_ci_last_good to file (only passing models)
    last_good_name = f"models_ci_last_good_{earliest}_to_{latest}.json"
    last_good_path = out_root / last_good_name
    logger.info(f"Writing last good passing models JSON to: {last_good_path}")
    last_good_text = json.dumps(models_ci_last_good, indent=2)
    last_good_path.write_text(last_good_text)
    logger.info(f"Wrote {len(last_good_text.encode('utf-8'))} bytes to {last_good_path}")
    
    # Log summary statistics
    total_models = len(all_models_dict)
    passing_models = len(models_ci_last_good)
    logger.info(f"Summary: {total_models} total models, {passing_models} passing models")
    

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Read On nightly CI results and summarize passing models")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--out-root", type=str, default="release_logs")
    parser.add_argument("--run-id", type=int, default=None, help="Process only this workflow run ID")
    parser.add_argument("--no-download", action="store_true", help="Process existing downloaded artifacts without re-downloading")
    parser.add_argument("--last-run-only", action="store_true", help="Process only the most recent run directory")
    args = parser.parse_args()
    
    # Setup output directory
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    logger.info(f"Output root directory: {out_root}")
    
    if not args.no_download:
        # Download mode: download artifacts first
        logger.info("=== DOWNLOAD MODE: Downloading artifacts ===")
        # Verify GitHub token authentication
        token = check_auth(args.owner, args.repo)
        download_runs(args.owner, args.repo, args.workflow_file, token, out_root, args.max_runs, args.run_id)
    
    logger.info("=== PROCESSING MODE: Processing existing artifacts ===")
    
    # Process all run directories
    logger.info("=== Processing all downloaded artifacts ===")
    all_models_dict, all_run_timestamps = process_all_runs(out_root, args.owner, args.repo, args.last_run_only)
    
    # Write output files
    write_summary_output(all_models_dict, all_run_timestamps, out_root)


if __name__ == "__main__":
    sys.exit(main())


