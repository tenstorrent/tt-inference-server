# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def resolve_commit(commit: str, repo_path: Path) -> Optional[str]:
    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        logger.warning(f"The path '{repo_path}' is not a valid directory.")
        return None
    try:
        _ = subprocess.run(["git", "cat-file", "-t", commit], cwd=str(repo_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rev = subprocess.run(["git", "rev-parse", commit], cwd=str(repo_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return rev.stdout.strip() if rev.returncode == 0 else None
    except Exception as exc:
        logger.warning(f"Error resolving commit '{commit}': {exc}")
        return None


def check_system_dependency(command: str) -> Dict[str, Any]:
    result = {"available": False, "version": None, "path": None, "error": None}
    try:
        path = shutil.which(command)
        if path:
            result["available"] = True
            result["path"] = path
            for version_cmd in [[command, "--version"], [command, "-V"], [command, "version"]]:
                try:
                    vr = subprocess.run(version_cmd, capture_output=True, text=True, timeout=10)
                    if vr.returncode == 0:
                        result["version"] = vr.stdout.strip().split("\n")[0]
                        break
                except Exception:
                    continue
    except Exception as exc:
        result["error"] = str(exc)
    return result


def get_required_system_dependencies() -> Dict[str, Dict[str, Any]]:
    required = [
        "git", "python3", "wget", "curl", "gcc", "jq", "vim", "htop", "screen", "tmux", "unzip", "zip", "rsync",
    ]
    return {dep: check_system_dependency(dep) for dep in required}


def get_commit_shas() -> Dict[str, Optional[str]]:
    commit_shas: Dict[str, Optional[str]] = {}
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if tt_metal_home and Path(tt_metal_home).exists():
        try:
            commit_shas["tt_metal"] = resolve_commit("HEAD", Path(tt_metal_home))
        except Exception as exc:
            logger.warning(f"Could not resolve tt-metal commit: {exc}")
            commit_shas["tt_metal"] = None
    else:
        commit_shas["tt_metal"] = None

    vllm_dir = os.getenv("vllm_dir")
    if vllm_dir and Path(vllm_dir).exists():
        try:
            commit_shas["vllm"] = resolve_commit("HEAD", Path(vllm_dir))
        except Exception as exc:
            logger.warning(f"Could not resolve vllm commit: {exc}")
            commit_shas["vllm"] = None
    else:
        commit_shas["vllm"] = None
    return commit_shas


def get_required_environment_vars() -> Dict[str, Optional[str]]:
    dockerfile_env_vars = [
        "TT_METAL_COMMIT_SHA_OR_TAG",
        "TT_VLLM_COMMIT_SHA_OR_TAG",
        "SHELL",
        "TZ",
        "ARCH_NAME",
        "TT_METAL_HOME",
        "CONFIG",
        "TT_METAL_ENV",
        "LOGURU_LEVEL",
        "PYTHONPATH",
        "PYTHON_ENV_DIR",
        "LD_LIBRARY_PATH",
        "vllm_dir",
        "VLLM_TARGET_DEVICE",
    ]

    runtime_env_vars = [
        "HF_MODEL_REPO_ID",
        "MODEL_IMPL",
        "MESH_DEVICE",
        "MODEL_SOURCE",
        "CACHE_ROOT",
        "MODEL_WEIGHTS_PATH",
        "TT_CACHE_PATH",
        "WH_ARCH_YAML",
        "LLAMA_DIR",
        "HF_MODEL",
        "HF_TOKEN",
        "JWT_SECRET",
        "VLLM_CONFIGURE_LOGGING",
        "VLLM_LOGGING_CONFIG",
        "VLLM_RPC_TIMEOUT",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "MAX_PREFILL_CHUNK_SIZE",
        "LLAMA_VERSION",
    ]

    vllm_config_vars = [
        "VLLM_BLOCK_SIZE",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_NUM_SCHEDULER_STEPS",
        "VLLM_MAX_LOG_LEN",
        "SERVICE_PORT",
        "VLLM_OVERRIDE_ARGS",
        "OVERRIDE_TT_CONFIG",
        "ENABLE_AUTO_TOOL_CHOICE",
    ]

    all_vars = dockerfile_env_vars + runtime_env_vars + vllm_config_vars
    return {var: os.getenv(var) for var in all_vars}


def get_python_environment_info() -> Dict[str, Any]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "python_path": os.getenv("PYTHONPATH"),
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }


def validate_preconditions(preconditions: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    critical_env_vars = ["HF_MODEL_REPO_ID", "TT_METAL_HOME", "CACHE_ROOT", "MODEL_WEIGHTS_PATH", "TT_CACHE_PATH"]
    for var in critical_env_vars:
        if not preconditions["environment_vars"].get(var):
            issues.append(f"Critical environment variable {var} is not set")

    path_vars = ["TT_METAL_HOME", "CACHE_ROOT", "MODEL_WEIGHTS_PATH", "TT_CACHE_PATH"]
    for var in path_vars:
        path_value = preconditions["environment_vars"].get(var)
        if path_value and not Path(path_value).exists():
            issues.append(f"Path for {var} does not exist: {path_value}")

    critical_deps = ["git", "python3", "gcc"]
    for dep in critical_deps:
        if not preconditions["system_dependencies"].get(dep, {}).get("available"):
            issues.append(f"Critical system dependency {dep} is not available")

    gcc_info = preconditions["system_dependencies"].get("gcc", {})
    if gcc_info.get("available") and gcc_info.get("version"):
        try:
            import re
            version_match = re.search(r"(\d+\.\d+\.\d+)", gcc_info["version"])
            if version_match:
                parts = [int(x) for x in version_match.group(1).split('.')]
                if parts < [6, 3, 0]:
                    issues.append(f"GCC version {version_match.group(1)} is below required 6.3.0+")
        except Exception as exc:
            logger.warning(f"Could not parse GCC version: {exc}")
    return issues


def validate_preconditions_file(json_file_path: Path) -> Dict[str, Any]:
    logger.info(f"Validating preconditions file: {json_file_path}")
    result: Dict[str, Any] = {
        "file_exists": False,
        "valid_json": False,
        "has_required_sections": False,
        "issues": [],
        "passed": False,
        "preconditions": None,
    }
    if not json_file_path.exists():
        result["issues"].append(f"Preconditions file does not exist: {json_file_path}")
        return result
    result["file_exists"] = True
    try:
        with open(json_file_path, "r") as f:
            pre = json.load(f)
        result["valid_json"] = True
        result["preconditions"] = pre
    except json.JSONDecodeError as exc:
        result["issues"].append(f"Invalid JSON format: {exc}")
        return result

    required_sections = [
        "timestamp",
        "system_info",
        "environment_vars",
        "commit_shas",
        "system_dependencies",
        "python_environment",
        "validation",
    ]
    missing = [s for s in required_sections if s not in pre]
    if missing:
        result["issues"].extend([f"Missing required section: {s}" for s in missing])
    else:
        result["has_required_sections"] = True

    result["issues"].extend(validate_preconditions(pre))
    if "validation" in pre and not pre["validation"].get("passed", False):
        result["issues"].append("Preconditions validation was recorded as failed during generation")
        if "issues" in pre["validation"]:
            result["issues"].extend([f"Recorded issue: {i}" for i in pre["validation"]["issues"]])

    result["passed"] = len(result["issues"]) == 0
    return result


def generate_preconditions_json(output_path: Optional[Path] = None) -> Dict[str, Any]:
    logger.info("Generating preconditions.json...")
    if output_path is None:
        output_path = Path("preconditions.json")

    pre = {
        "timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        "system_info": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "environment_vars": get_required_environment_vars(),
        "commit_shas": get_commit_shas(),
        "system_dependencies": get_required_system_dependencies(),
        "python_environment": {
            "version": sys.version,
            "executable": sys.executable,
            "virtual_env": os.getenv("VIRTUAL_ENV"),
            "python_path": os.getenv("PYTHONPATH"),
        },
    }

    issues = validate_preconditions(pre)
    pre["validation"] = {"passed": len(issues) == 0, "issues": issues}

    with open(output_path, "w") as f:
        json.dump(pre, f, indent=2, default=str)
    logger.info(f"Preconditions saved to: {output_path}")

    if issues:
        logger.warning(f"Validation found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All preconditions validation checks passed")
    return pre


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Generate and validate preconditions.json")
    parser.add_argument("--validate", metavar="FILE", type=Path, help="Validate an existing preconditions.json file")
    parser.add_argument("--output", "-o", metavar="FILE", type=Path, help="Output path for generated preconditions.json")
    args = parser.parse_args()

    try:
        if args.validate:
            res = validate_preconditions_file(args.validate)
            logger.info("=== Preconditions Validation Results ===")
            logger.info(f"File exists: {res['file_exists']}")
            logger.info(f"Valid JSON: {res['valid_json']}")
            logger.info(f"Has required sections: {res['has_required_sections']}")
            logger.info(f"Validation passed: {res['passed']}")
            if res["issues"]:
                logger.warning(f"Found {len(res['issues'])} validation issues:")
                for issue in res["issues"]:
                    logger.warning(f"  - {issue}")
            if not res["passed"]:
                logger.error("Preconditions validation failed.")
                sys.exit(1)
            else:
                logger.info("Preconditions validation passed successfully.")
        else:
            pre = generate_preconditions_json(args.output)
            logger.info("=== Preconditions Summary ===")
            logger.info(f"Environment variables collected: {len(pre['environment_vars'])}")
            logger.info(f"System dependencies checked: {len(pre['system_dependencies'])}")
            logger.info(f"Commit SHAs resolved: {sum(1 for sha in pre['commit_shas'].values() if sha)}")
            logger.info(f"Validation passed: {pre['validation']['passed']}")
            if not pre["validation"]["passed"]:
                logger.error("Preconditions validation failed. Check the issues above.")
                sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to process preconditions: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


