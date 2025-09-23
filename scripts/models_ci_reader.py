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


def http_get(url: str, token: str, accept: Optional[str] = None, retry: int = 3, timeout: int = 60) -> bytes:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "models-ci-reader/1.0",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if accept:
        headers["Accept"] = accept
    logger.info(f"HTTP GET {url}")
    logger.debug(f"Re-run with: {_curl_debug_string(url, accept)}")
    for attempt in range(retry):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
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
    data = http_get(url, token, accept="application/vnd.github+json", retry=retry, timeout=60)
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


def download_run_logs_zip(run_id: int, owner: str, repo: str, token: str) -> bytes:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    try:
        # GitHub API generally expects application/vnd.github+json; it will 302 to the ZIP
        return http_get(url, token, accept="application/vnd.github+json", timeout=300)
    except HTTPError as e:
        if e.code in (406, 415):
            logger.info("Retrying run logs download without Accept header due to HTTP error")
            return http_get(url, token, accept=None, timeout=300)
        raise


def download_artifact_zip(artifact_ref, owner: str, repo: str, token: str) -> bytes:
    """Download an artifact ZIP by id or by full archive_download_url with robust fallbacks."""
    # Allow passing in archive_download_url directly
    if isinstance(artifact_ref, str) and artifact_ref.startswith("http"):
        url = artifact_ref
    else:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts/{artifact_ref}/zip"

    # Try multiple Accept headers to accommodate API behaviors
    accept_order = [
        "application/zip",
        "application/octet-stream",
        "application/vnd.github+json",
        None,
    ]

    last_error: Optional[Exception] = None
    for accept in accept_order:
        try:
            return http_get(url, token, accept=accept, retry=3, timeout=300)
        except HTTPError as e:
            last_error = e
            # For expired/unavailable artifacts, bubble up after trying fallbacks
            if e.code in (400, 404, 410):
                logger.warning(
                    f"Artifact download failed with HTTP {e.code} for {url}. This artifact may be expired or unavailable."
                )
                continue
            # Retry loop will try next Accept variant
            continue
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Failed to download artifact ZIP: unknown error")


def ensure_dir(path: Path) -> None:
    before_exists = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    if before_exists:
        logger.debug(f"Directory already exists: {path}")
    else:
        logger.info(f"Created directory: {path}")


def extract_zip_to_dir(zip_bytes: bytes, out_dir: Path) -> None:
    ensure_dir(out_dir)
    logger.info(f"Extracting zip to: {out_dir}")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)
    logger.info(f"Extracted files to: {out_dir}")


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
    logger.debug(f"Globbing for pattern '{pattern}' in directory: {dir_path}")
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
        logger.info(f"Reading model spec JSON: {spec_file}")
        data = json.loads(spec_file.read_text())
    except Exception:
        return None, None
    model_id = data.get("model_id")
    return data, model_id


def process_artifact_dir(artifact_dir: Path) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    # Returns (model_spec_json, report_data_json, model_id)
    logger.info(f"Processing artifact directory: {artifact_dir}")
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
                        logger.info(f"Reading report data JSON: {report_file}")
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


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Read On nightly CI results and summarize passing models")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--out-root", type=str, default=f"models_ci_reader_output_{timestamp}")
    parser.add_argument("--run-id", type=int, default=None, help="Process only this workflow run ID")
    args = parser.parse_args()

    # Step 1: Verify GitHub token authentication FIRST
    token = check_auth(args.owner, args.repo)

    # Step 2: Setup output directory
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    logger.info(f"Output root directory: {out_root}")

    # Step 3: Resolve workflow and list recent runs
    wf = get_workflow(args.owner, args.repo, args.workflow_file, token)
    workflow_id = wf.get("id")
    assert workflow_id, "Could not resolve workflow id"

    runs: List[dict]
    if args.run_id is not None:
        logger.info(f"Fetching single workflow run by ID: {args.run_id}")
        run_obj = get_workflow_run(args.run_id, args.owner, args.repo, token)
        # Validate this run belongs to the requested workflow file
        run_workflow_file = (run_obj.get("name") or "").lower()
        # We cannot always rely on 'path' here; still proceed with logging the workflow name
        logger.info(f"Fetched run {args.run_id} (workflow name: '{run_workflow_file}')")
        runs = [run_obj]
    else:
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
        logger.info(f"Run output directory: {run_out_dir}")

        # Download run logs and parse build-inference-server args
        test_tt_metal_commit: Optional[str] = None
        test_vllm_commit: Optional[str] = None
        try:
            logs_zip = download_run_logs_zip(run_id, args.owner, args.repo, token)
            logs_dir = run_out_dir / "logs"
            if logs_dir.exists():
                logger.info(f"Removing existing logs directory: {logs_dir}")
                shutil.rmtree(logs_dir)
            extract_zip_to_dir(logs_zip, logs_dir)
            # Extract commit shas from build log format
            test_tt_metal_commit, test_vllm_commit = find_commits_from_logs(logs_dir)
            # Extract runner names per job; pick the build job runner if available
            runner_names = parse_runner_names(logs_dir)
            build_runner_name: Optional[str] = None
            for job_dir_name, runner_name in runner_names.items():
                if "build-inference-server" in job_dir_name.lower():
                    build_runner_name = runner_name
                    break
        except Exception:
            runner_names = {}
            build_runner_name = None

        # List and download artifacts matching workflow logs
        try:
            artifacts = list_run_artifacts(run_id, args.owner, args.repo, token)
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
                logger.debug(f"Skipping artifact without id or download url: {name}")
                continue
            try:
                ref = archive_url if archive_url else artifact_id
                z = download_artifact_zip(ref, args.owner, args.repo, token)
                # Extract under run directory in folder named as artifact name
                art_dir = run_out_dir / name
                if art_dir.exists():
                    logger.info(f"Removing existing artifact directory: {art_dir}")
                    shutil.rmtree(art_dir)
                extract_zip_to_dir(z, art_dir)
            except Exception as e:
                logger.warning(f"Failed to download/extract artifact {name}: {e}")
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
                    "build_runner": build_runner_name,
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
    logger.info(f"Writing summary JSON to: {summary_path}")
    summary_text = json.dumps(passing_dict, indent=2)
    summary_path.write_text(summary_text)
    logger.info(f"Wrote {len(summary_text.encode('utf-8'))} bytes to {summary_path}")

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


