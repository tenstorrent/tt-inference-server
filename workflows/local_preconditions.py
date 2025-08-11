# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def resolve_commit(commit: str, repo_path: Path) -> Optional[str]:
    """Resolve a commit reference to its full SHA for a given repository path.

    Returns None if the path is invalid or the ref cannot be resolved.
    """
    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        logger.warning(f"The path '{repo_path}' is not a valid directory.")
        return None

    try:
        type_check = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if type_check.stdout.strip() not in ["commit", "tag"]:
            logger.warning(f"Commit '{commit}' is not in the repository.")
            return None

        rev_parse = subprocess.run(
            ["git", "rev-parse", commit],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return rev_parse.stdout.strip() if rev_parse.returncode == 0 else None
    except Exception as exc:
        logger.warning(f"Error resolving commit '{commit}': {exc}")
        return None


def categorize_environment_variables(env_vars: Dict[str, str]) -> Dict[str, List[str]]:
    """Group environment variables into debugging-friendly categories."""
    categories: Dict[str, List[str]] = {
        "tt_inference_system": [],
        "development_environment": [],
        "container_runtime": [],
        "system_core": [],
        "authentication_secrets": [],
        "other_applications": [],
    }

    for var_name in env_vars.keys():
        var_upper = var_name.upper()

        if any(p in var_upper for p in [
            "TT_", "VLLM_", "HF_", "MODEL_", "LLAMA_", "MESH_", "ARCH_",
            "CACHE_", "WH_", "INFERENCE_", "GPU_", "CUDA_",
        ]):
            categories["tt_inference_system"].append(var_name)
        elif (
            any(p in var_upper for p in [
                "PYTHON", "PIP_", "CONDA_", "VIRTUAL_", "VENV", "_ENV",
                "GCC_", "CC_", "CXX_", "CMAKE_", "PKG_CONFIG", "LIBRARY_",
                "INCLUDE_", "LD_", "LOGURU_", "CONFIG",
            ]) or var_name in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]
        ):
            categories["development_environment"].append(var_name)
        elif (
            any(p in var_upper for p in ["DOCKER_", "CONTAINER_", "SERVICE_", "PORT", "HOSTNAME"]) or
            var_name in ["TZ", "DEBIAN_FRONTEND"]
        ):
            categories["container_runtime"].append(var_name)
        elif (
            any(p in var_upper for p in ["TOKEN", "SECRET", "KEY", "PASSWORD", "AUTH", "CREDENTIAL", "JWT_"]) and
            not any(ex in var_upper for ex in ["BATCHED_TOKENS", "MAX_TOKENS", "NUM_TOKENS"])
        ):
            categories["authentication_secrets"].append(var_name)
        elif any(p in var_upper for p in ["HOME", "USER", "SHELL", "TERM", "LANG", "LC_", "PWD", "OLDPWD"]) or var_name in ["SHELL", "HOME", "USER", "LOGNAME", "TERM"]:
            categories["system_core"].append(var_name)
        else:
            categories["other_applications"].append(var_name)

    return {k: v for k, v in categories.items() if v}


def filter_sensitive_variables(env_vars: Dict[str, str], include_sensitive: bool = False) -> Dict[str, Any]:
    """Redact sensitive variables while preserving configuration values.

    If include_sensitive is True, returns env_vars unchanged.
    """
    if include_sensitive:
        return env_vars

    sensitive_exact = {"HF_TOKEN", "JWT_SECRET"}
    sensitive_patterns = ["PASSWORD", "SECRET", "API_KEY", "_KEY", "AUTH_TOKEN", "ACCESS_TOKEN", "CREDENTIAL"]
    config_exclusions = ["BATCHED_TOKENS", "MAX_TOKENS", "NUM_TOKENS"]

    filtered: Dict[str, Any] = {}
    sensitive_count = 0

    for name, value in env_vars.items():
        upper = name.upper()
        is_sensitive = name in sensitive_exact
        if not is_sensitive:
            is_config = any(ex in upper for ex in config_exclusions)
            if not is_config:
                is_sensitive = any(pat in upper for pat in sensitive_patterns)

        if is_sensitive:
            sensitive_count += 1
            filtered[name] = "<REDACTED>" if value else None
        else:
            filtered[name] = value

    if sensitive_count:
        logger.info(f"Filtered {sensitive_count} sensitive variables")
    return filtered


def get_environment_analysis(include_sensitive: bool = False) -> Dict[str, Any]:
    """Collect, optionally redact, and categorize environment variables."""
    env_vars = dict(os.environ)
    if not include_sensitive:
        env_vars = filter_sensitive_variables(env_vars, include_sensitive)

    categories = categorize_environment_variables(env_vars)
    return {
        "all_environment": env_vars,
        "categories": categories,
        "statistics": {
            "total_variables": len(env_vars),
            "category_breakdown": {k: len(v) for k, v in categories.items()},
            "sensitive_filtered": not include_sensitive,
        },
    }


def get_commit_information() -> Dict[str, Optional[str]]:
    """Resolve HEAD commit SHAs for TT-Metal and vLLM if directories are present."""
    commit_info: Dict[str, Optional[str]] = {}

    tt_metal_home = os.getenv("TT_METAL_HOME")
    if tt_metal_home and Path(tt_metal_home).exists():
        try:
            commit_info["tt_metal"] = resolve_commit("HEAD", Path(tt_metal_home))
        except Exception as exc:
            logger.warning(f"Could not resolve tt-metal commit: {exc}")
            commit_info["tt_metal"] = None
    else:
        commit_info["tt_metal"] = None

    vllm_dir = os.getenv("vllm_dir")
    if vllm_dir and Path(vllm_dir).exists():
        try:
            commit_info["vllm"] = resolve_commit("HEAD", Path(vllm_dir))
        except Exception as exc:
            logger.warning(f"Could not resolve vllm commit: {exc}")
            commit_info["vllm"] = None
    else:
        commit_info["vllm"] = None

    return commit_info


def get_run_command_reconstruction() -> Dict[str, Any]:
    """Infer an approximate run.py command from environment variables."""
    reconstructed: List[str] = []
    evidence: List[str] = []

    model_repo = os.getenv("HF_MODEL_REPO_ID")
    if model_repo:
        model_name = model_repo.split("/")[-1] if "/" in model_repo else model_repo
        reconstructed.append(f"--model {model_name}")
        evidence.append("HF_MODEL_REPO_ID")

    mesh_device = os.getenv("MESH_DEVICE")
    if mesh_device:
        reconstructed.append(f"--device {mesh_device.lower()}")
        evidence.append("MESH_DEVICE")

    container_indicators = [
        os.getenv("CONTAINER_APP_USERNAME"),
        os.getenv("HOSTNAME", "").startswith(("docker", "container")) or len(os.getenv("HOSTNAME", "")) == 12,
        os.path.exists("/.dockerenv"),
        os.getenv("SERVICE_PORT"),
    ]
    if any(container_indicators):
        reconstructed.extend(["--workflow server", "--docker-server"])
        evidence.extend(["container_detection", "SERVICE_PORT"])

    model_impl = os.getenv("MODEL_IMPL")
    if model_impl and model_impl != "tt-transformers":
        reconstructed.append(f"--impl {model_impl}")
        evidence.append("MODEL_IMPL")

    if os.getenv("TT_METAL_ENV") == "dev":
        reconstructed.append("--dev-mode")
        evidence.append("TT_METAL_ENV")

    service_port = os.getenv("SERVICE_PORT")
    if service_port and service_port != "8000":
        reconstructed.append(f"--service-port {service_port}")
        evidence.append("SERVICE_PORT")

    if os.getenv("OVERRIDE_TT_CONFIG"):
        override_config = os.getenv("OVERRIDE_TT_CONFIG")
        reconstructed.append(f"--override-tt-config '{override_config}'")
        evidence.append("OVERRIDE_TT_CONFIG")

    vllm_overrides: Dict[str, str] = {}
    defaults = {"VLLM_MAX_MODEL_LEN": "131072", "VLLM_MAX_NUM_SEQS": "32"}
    for env_var, default_val in defaults.items():
        value = os.getenv(env_var)
        if value and value != default_val:
            key = env_var.replace("VLLM_", "").lower()
            vllm_overrides[key] = value
            evidence.append(env_var)

    if vllm_overrides:
        reconstructed.append(f"--vllm-override-args '{json.dumps(vllm_overrides)}'")

    return {
        "command": f"python3 run.py {' '.join(reconstructed)}" if reconstructed else None,
        "arguments": reconstructed,
        "evidence_sources": evidence,
    }


def _extract_container_id_from_cgroup() -> Optional[str]:
    if not os.path.exists("/proc/self/cgroup"):
        return None
    try:
        with open("/proc/self/cgroup", "r") as f:
            content = f.read()
        patterns = [
            r"/docker/([a-f0-9]{64})",
            r"docker-([a-f0-9]{64})",
            r"/([a-f0-9]{64})\.scope",
            r"containers\.slice/docker-([a-f0-9]{64})",
        ]
        for pat in patterns:
            match = re.search(pat, content)
            if match:
                return match.group(1)[:12]
    except Exception as exc:
        logger.debug(f"Error reading cgroup: {exc}")
    return None


def _extract_container_id_from_mountinfo() -> Optional[str]:
    if not os.path.exists("/proc/self/mountinfo"):
        return None
    try:
        with open("/proc/self/mountinfo", "r") as f:
            content = f.read()
        if "docker" in content.lower():
            match = re.search(r"/var/lib/docker/containers/([a-f0-9]{64})", content)
            if match:
                return match.group(1)[:12]
    except Exception as exc:
        logger.debug(f"Error reading mountinfo: {exc}")
    return None


def _get_container_info_via_docker(container_id: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if not shutil.which("docker"):
        return info
    try:
        result = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            inspect = json.loads(result.stdout)[0]
            config = inspect.get("Config", {})
            if config.get("Image"):
                info["image_name"] = config["Image"]
            if inspect.get("Image"):
                info["image_id"] = inspect["Image"]
            if inspect.get("Created"):
                info["created"] = inspect["Created"]
            info["extraction_method"] = "docker_inspect_success"
    except Exception as exc:
        logger.debug(f"Could not inspect container: {exc}")
    return info


def get_docker_container_info() -> Dict[str, Any]:
    """Return container metadata if running in Docker; otherwise indicate not in a container."""
    indicators = [
        os.path.exists("/.dockerenv"),
        os.getenv("CONTAINER_APP_USERNAME"),
        os.getenv("HOSTNAME", "").startswith(("docker", "container")) or len(os.getenv("HOSTNAME", "")) == 12,
    ]
    if not any(indicators):
        return {"in_container": False}

    info: Dict[str, Any] = {"in_container": True}
    try:
        container_id = _extract_container_id_from_cgroup() or _extract_container_id_from_mountinfo()
        if not container_id:
            hostname = os.getenv("HOSTNAME", "")
            if hostname and len(hostname) == 12 and re.match(r"^[a-f0-9]{12}$", hostname):
                container_id = hostname
        if container_id:
            info["container_id"] = container_id
            info["extraction_method"] = "cgroup"
            info.update(_get_container_info_via_docker(container_id))
        if "extraction_method" not in info:
            info["extraction_method"] = "dockerenv_file" if os.path.exists("/.dockerenv") else "environment_indicators"
        if "image_name" not in info:
            for var in ["IMAGE_NAME", "DOCKER_IMAGE", "CONTAINER_IMAGE"]:
                val = os.getenv(var)
                if val:
                    info["image_name"] = val
                    break
    except Exception as exc:
        logger.warning(f"Error extracting Docker container info: {exc}")
    return info


def get_container_dependencies() -> Dict[str, Any]:
    """Return output of pip freeze if available (useful inside container)."""
    deps = {"python_packages": {}, "python_packages_count": 0, "extraction_available": False}
    try:
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            pkgs: Dict[str, str] = {}
            for line in result.stdout.strip().split("\n"):
                if not line or "==" not in line:
                    continue
                try:
                    name, version = line.split("==", 1)
                except ValueError:
                    pkgs[line] = "unknown"
                else:
                    pkgs[name] = version
            deps.update({
                "python_packages": pkgs,
                "python_packages_count": len(pkgs),
                "extraction_available": True,
            })
    except Exception as exc:
        logger.debug(f"Could not get pip packages: {exc}")
    return deps


def check_system_dependency(command: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "version": None, "path": None, "error": None}
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


def get_system_dependencies() -> Dict[str, Dict[str, Any]]:
    critical = [
        "gcc", "git", "python3", "curl", "wget", "cmake", "make", "jq", "vim", "unzip", "zip", "rsync", "docker",
    ]
    return {dep: check_system_dependency(dep) for dep in critical}


def generate_preconditions_json(output_path: Optional[Path] = None, include_sensitive: bool = False) -> Dict[str, Any]:
    """Generate comprehensive preconditions for CI/debug reproduction.

    If output_path is provided, the JSON file will be saved there.
    """
    logger.info("Generating CI preconditions extraction...")

    env_analysis = get_environment_analysis(include_sensitive)
    commit_info = get_commit_information()
    command_reconstruction = get_run_command_reconstruction()
    docker_info = get_docker_container_info()
    container_deps = get_container_dependencies()
    system_deps = get_system_dependencies()

    filtered_env_vars = env_analysis["all_environment"]
    sensitive_count = sum(1 for v in filtered_env_vars.values() if v == "<REDACTED>")

    environment_vars: Dict[str, Any] = {
        "statistics": env_analysis["statistics"],
        "sensitive_filtered_count": sensitive_count,
    }
    for category, var_names in env_analysis["categories"].items():
        environment_vars[category] = {var: env_analysis["all_environment"].get(var) for var in var_names}

    preconditions: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment_vars": environment_vars,
        "commit_shas": commit_info,
        "run_command_reconstruction": command_reconstruction,
        "docker_container_info": docker_info,
        "system_dependencies": system_deps,
        "container_dependencies": container_deps,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving CI preconditions to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(preconditions, f, indent=2)

    return preconditions


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Extract CI preconditions for Docker container reproduction")
    parser.add_argument("--output", "-o", metavar="FILE", type=Path, help="Output path for generated preconditions.json")
    parser.add_argument("--include-sensitive", action="store_true", help="Include sensitive env vars (tokens, keys, etc.)")

    args = parser.parse_args()

    try:
        pre = generate_preconditions_json(args.output, include_sensitive=args.include_sensitive)
        logger.info("=== Complete Environment Extraction Summary ===")
        stats = pre["environment_vars"]["statistics"]
        logger.info(f"Total environment variables: {stats['total_variables']}")
        logger.info("Category breakdown:")
        for cat, cnt in stats["category_breakdown"].items():
            logger.info(f"  - {cat}: {cnt} variables")
        commit_resolved = sum(1 for sha in pre["commit_shas"].values() if sha)
        logger.info(f"Commit SHAs resolved: {commit_resolved}")
        logger.info(f"Run command extracted: {bool(pre['run_command_reconstruction'].get('command'))}")
        logger.info(f"In Docker container: {pre['docker_container_info']['in_container']}")
        sys_deps = pre["system_dependencies"]
        available = sum(1 for dep in sys_deps.values() if dep.get("available"))
        logger.info(f"System dependencies: {available}/{len(sys_deps)} available")
        logger.info(f"Python packages extracted: {pre['container_dependencies']['python_packages_count']}")
    except Exception as exc:
        logger.error(f"Failed to extract preconditions: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


