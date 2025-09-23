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


def http_get(url: str, token: str, accept: Optional[str] = None, github_api_version: str = "2022-11-28", retry: int = 3) -> bytes:
    logger.info(f"Making HTTP GET request to: {url}")
    headers = {
        "Authorization": f"Bearer {token[:10]}...",
        "User-Agent": "last-known-good-model-runs/1.0",
        "X-GitHub-Api-Version": github_api_version
    }
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
            if e.code == 401:
                # Extract owner/repo from URL
                owner_repo = url.replace("https://api.github.com/repos/", "").split("/")[0:2]
                repo_path = "/".join(owner_repo) if len(owner_repo) >= 2 else "unknown"
                logger.error("❌ AUTHENTICATION FAILED: GitHub token lacks access to private repository")
                logger.error(f"   Repository: {repo_path}")
                logger.error("   Required: GitHub token with 'repo' scope or fine-grained token with repository access")
                logger.error("   Solution: Create new GitHub token at https://github.com/settings/tokens")
                logger.error("   Current token: Check if it has 'repo' scope for private repositories")
            elif e.code == 404:
                owner_repo = url.replace("https://api.github.com/repos/", "").split("/")[0:2]
                repo_path = "/".join(owner_repo) if len(owner_repo) >= 2 else "unknown"
                logger.error(f"❌ NOT FOUND: Repository or resource does not exist or is not accessible")
                logger.error(f"   Repository: {repo_path}")
                logger.error("   Solution: Verify repository exists and token has access")
            elif e.code in (429, 500, 502, 503, 504) and attempt < retry - 1:
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


def http_json(url: str, token: str, github_api_version: str = "2022-11-28", retry: int = 3) -> dict:
    data = http_get(url, token, accept="application/vnd.github+json", github_api_version=github_api_version, retry=retry)
    return json.loads(data.decode("utf-8"))


def list_workflows(owner: str, repo: str, token: str, github_api_version: str = "2022-11-28") -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows"
    data = http_json(url, token, github_api_version=github_api_version)
    return data.get("workflows", [])


def get_workflow(owner: str, repo: str, workflow_file: str, token: str, github_api_version: str = "2022-11-28") -> dict:
    # List workflows and match by name or path suffix
    workflows = list_workflows(owner, repo, token, github_api_version=github_api_version)
    for wf in workflows:
        path = wf.get("path", "")
        name = wf.get("name", "")
        if path.endswith(f"/{workflow_file}"):
            return wf
        if name.lower() == "on nightly":
            return wf
    raise HTTPError(f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows", 404, f"Workflow {workflow_file} not found", hdrs=None, fp=None)


def list_workflow_runs(workflow_id: int, owner: str, repo: str, token: str, per_page: int = 30, github_api_version: str = "2022-11-28") -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs?per_page={per_page}"
    data = http_json(url, token, github_api_version=github_api_version)
    return data.get("workflow_runs", [])


def list_repo_runs(owner: str, repo: str, token: str, per_page: int = 100, github_api_version: str = "2022-11-28") -> List[dict]:
    # Get all workflows first
    workflows = list_workflows(owner, repo, token, github_api_version=github_api_version)
    all_runs = []

    # For each workflow, get its runs
    for workflow in workflows:
        workflow_id = workflow.get("id")
        if workflow_id:
            try:
                runs = list_workflow_runs(workflow_id, owner, repo, token, per_page=per_page, github_api_version=github_api_version)
                all_runs.extend(runs)
            except Exception as e:
                logger.warning(f"Failed to get runs for workflow {workflow.get('name', 'unknown')}: {e}")
                continue

    return all_runs


def list_run_artifacts(run_id: int, owner: str, repo: str, token: str, github_api_version: str = "2022-11-28") -> List[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    data = http_json(url, token, github_api_version=github_api_version)
    return data.get("artifacts", [])


def download_run_logs_zip(run_id: int, owner: str, repo: str, token: str, github_api_version: str = "2022-11-28") -> bytes:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    return http_get(url, token, accept="application/zip", github_api_version=github_api_version)


def download_artifact_zip(artifact_id: int, owner: str, repo: str, token: str, github_api_version: str = "2022-11-28") -> bytes:
    # Endpoint requires "/zip"
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
    return http_get(url, token, accept="application/zip", github_api_version=github_api_version)


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
    logger.info("🔐 Checking GitHub token authentication...")

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
        "User-Agent": "last-known-good-model-runs/1.0",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
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
    parser = argparse.ArgumentParser(description="Find last known good model runs from On nightly CI")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--out-root", type=str, default=".")
    args = parser.parse_args()

    logger.info("🚀 Starting last_known_good_model_runs.py script")
    logger.info(f"   Arguments: owner={args.owner}, repo={args.repo}, workflow_file={args.workflow_file}, max_runs={args.max_runs}")

    # Step 1: Verify GitHub token authentication FIRST
    logger.info("📋 Step 1: Authentication check")
    token = check_auth(args.owner, args.repo)

    # Step 2: Setup output directory
    logger.info("📁 Step 2: Setting up output directory")
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    logger.info(f"   Output directory: {out_root}")
    logger.info("✅ Authentication and setup complete")
    logger.info("")

    # Step 3: Find and process workflow runs
    logger.info("🔍 Step 3: Finding workflow runs")
    runs: List[dict] = []  # This will be populated with workflow run data
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
        if e.code == 401:
            logger.error("❌ WORKFLOW API AUTHENTICATION FAILED")
            logger.error(f"   Cannot access workflow data for {args.owner}/{args.repo}")
            logger.error("   This is likely due to insufficient GitHub token permissions")
        else:
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
        except HTTPError as e:
            if e.code == 401:
                logger.error(f"❌ ARTIFACTS API AUTHENTICATION FAILED for run {run_id}")
                logger.error("   Cannot access artifacts due to insufficient GitHub token permissions")
                logger.error(f"   Repository: {args.owner}/{args.repo}")
            else:
                logger.warning(f"Failed to list artifacts for run {run_id}: {e.code} {e.reason}")
            artifacts = []
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

    # Step 4: Serialize passing_dict to JSON file
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

    # Step 5: Print latest passing values per model_id (without model_spec_json)
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

    # Step 6: Final completion summary
    logger.info("")
    logger.info("🎉 Script completed successfully!")
    logger.info("📊 Summary:")
    logger.info(f"   • Processed {len(runs)} workflow runs")
    logger.info(f"   • Found {len(passing_dict)} models with passing results")
    logger.info(f"   • Generated summary: {summary_path}")
    logger.info(f"   • Output directory: {out_root}")


if __name__ == "__main__":
    sys.exit(main())
