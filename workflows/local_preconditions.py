# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import datetime
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def resolve_commit(commit: str, repo_path: Path) -> str:
    """Resolve a commit reference to its full SHA.
    
    Args:
        commit: Commit reference (e.g., 'HEAD', tag, or SHA)
        repo_path: Path to the git repository
        
    Returns:
        Full commit SHA string, or None if not found
    """
    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        logger.warning(f"The path '{repo_path}' is not a valid directory.")
        return None
    
    try:
        # Check if the commit/tag exists
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout.strip() not in ["commit", "tag"]:
            logger.warning(f"Commit '{commit}' is not in the repository.")
            return None

        # Get full SHA
        result = subprocess.run(
            ["git", "rev-parse", commit],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        logger.warning(f"Error resolving commit '{commit}': {e}")
        return None


def categorize_environment_variables(env_vars: Dict[str, str]) -> Dict[str, List[str]]:
    """Categorize environment variables into logical debugging categories.
    
    Groups variables by their purpose to aid in debugging and reproduction.
    
    Args:
        env_vars: Dictionary of all environment variables
        
    Returns:
        Dict with categorized variables (empty categories are removed)
    """
    categories = {
        "tt_inference_system": [],      # TT-Metal, vLLM, models, inference-specific
        "development_environment": [], # Python, development tools, paths
        "container_runtime": [],       # Docker, container-specific variables  
        "system_core": [],            # Core OS, shell, user environment
        "authentication_secrets": [], # Tokens, keys, credentials
        "other_applications": []      # Everything else
    }
    
    for var_name in env_vars.keys():
        var_upper = var_name.upper()
        
        # TT-Inference System: Hardware, models, inference frameworks
        if any(pattern in var_upper for pattern in [
            'TT_', 'VLLM_', 'HF_', 'MODEL_', 'LLAMA_', 'MESH_', 'ARCH_',
            'CACHE_', 'WH_', 'INFERENCE_', 'GPU_', 'CUDA_'
        ]):
            categories["tt_inference_system"].append(var_name)
        
        # Development Environment: Python, build tools, development
        elif any(pattern in var_upper for pattern in [
            'PYTHON', 'PIP_', 'CONDA_', 'VIRTUAL_', 'VENV', '_ENV',
            'GCC_', 'CC_', 'CXX_', 'CMAKE_', 'PKG_CONFIG', 'LIBRARY_',
            'INCLUDE_', 'LD_', 'LOGURU_', 'CONFIG'
        ]) or var_name in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']:
            categories["development_environment"].append(var_name)
        
        # Container Runtime: Docker, services, container-specific
        elif any(pattern in var_upper for pattern in [
            'DOCKER_', 'CONTAINER_', 'SERVICE_', 'PORT', 'HOSTNAME'
        ]) or var_name in ['TZ', 'DEBIAN_FRONTEND']:
            categories["container_runtime"].append(var_name)
        
        # Authentication & Secrets: Tokens, keys, passwords (excluding config parameters)
        elif any(pattern in var_upper for pattern in [
            'TOKEN', 'SECRET', 'KEY', 'PASSWORD', 'AUTH', 'CREDENTIAL', 'JWT_'
        ]) and not any(exclusion in var_upper for exclusion in [
            'BATCHED_TOKENS', 'MAX_TOKENS', 'NUM_TOKENS'  # Config, not auth
        ]):
            categories["authentication_secrets"].append(var_name)
        
        # System Core: OS, shell, user, basic system
        elif any(pattern in var_upper for pattern in [
            'HOME', 'USER', 'SHELL', 'TERM', 'LANG', 'LC_', 'PWD', 'OLDPWD'
        ]) or var_name in ['SHELL', 'HOME', 'USER', 'LOGNAME', 'TERM']:
            categories["system_core"].append(var_name)
        
        # Everything else
        else:
            categories["other_applications"].append(var_name)
    
    return {k: v for k, v in categories.items() if v}


def filter_sensitive_variables(env_vars: Dict[str, str], 
                             include_sensitive: bool = False) -> Dict[str, str]:
    """Filter out sensitive environment variables for security.
    
    Redacts authentication-related variables while preserving configuration parameters.
    
    Args:
        env_vars: Dictionary of environment variables
        include_sensitive: Whether to include sensitive variables (default: False)
        
    Returns:
        Filtered dictionary with sensitive variables redacted as "<REDACTED>"
    """
    if include_sensitive:
        return env_vars
    
    # Known sensitive variables
    sensitive_exact_matches = ['HF_TOKEN', 'JWT_SECRET']
    
    # Generic patterns for sensitive variables
    sensitive_patterns = [
        'PASSWORD', 'SECRET', 'API_KEY', '_KEY', 'AUTH_TOKEN', 'ACCESS_TOKEN', 'CREDENTIAL'
    ]
    
    # Configuration parameters that are not authentication-related
    config_exclusions = ['BATCHED_TOKENS', 'MAX_TOKENS', 'NUM_TOKENS']
    
    filtered_vars = {}
    sensitive_count = 0
    
    for var_name, var_value in env_vars.items():
        var_upper = var_name.upper()
        
        # Check if it's a known sensitive variable
        is_sensitive = var_name in sensitive_exact_matches
        
        # Check patterns but exclude config variables
        if not is_sensitive:
            is_config = any(exclusion in var_upper for exclusion in config_exclusions)
            if not is_config:
                is_sensitive = any(pattern in var_upper for pattern in sensitive_patterns)
        
        if is_sensitive:
            sensitive_count += 1
            filtered_vars[var_name] = "<REDACTED>" if var_value else None
        else:
            filtered_vars[var_name] = var_value
    
    if sensitive_count > 0:
        logger.info(f"Filtered {sensitive_count} sensitive variables")
    
    return filtered_vars


def get_environment_analysis(include_sensitive: bool = False) -> Dict[str, Any]:
    """Get comprehensive analysis of all environment variables.
    
    Args:
        include_sensitive: Whether to include sensitive variables (default: False)
        
    Returns:
        Dict with environment variables, categorization, and statistics
    """
    all_env_vars = dict(os.environ)
    
    if not include_sensitive:
        all_env_vars = filter_sensitive_variables(all_env_vars, include_sensitive)
    
    categories = categorize_environment_variables(all_env_vars)
    
    return {
        "all_environment": all_env_vars,
        "categories": categories,
        "statistics": {
            "total_variables": len(all_env_vars),
            "category_breakdown": {k: len(v) for k, v in categories.items()},
            "sensitive_filtered": not include_sensitive
        }
    }


def get_commit_information() -> Dict[str, Optional[str]]:
    """Get commit SHAs for relevant repositories.
    
    Returns:
        Dict with commit SHAs for TT-Metal and vLLM repositories if available
    """
    commit_info = {}
    
    # Check for TT-Metal repository
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if tt_metal_home and Path(tt_metal_home).exists():
        try:
            commit_info["tt_metal"] = resolve_commit("HEAD", Path(tt_metal_home))
        except Exception as e:
            logger.warning(f"Could not resolve tt-metal commit: {e}")
            commit_info["tt_metal"] = None
    else:
        commit_info["tt_metal"] = None
        
    # Check for vLLM repository
    vllm_dir = os.getenv("vllm_dir")
    if vllm_dir and Path(vllm_dir).exists():
        try:
            commit_info["vllm"] = resolve_commit("HEAD", Path(vllm_dir))
        except Exception as e:
            logger.warning(f"Could not resolve vllm commit: {e}")
            commit_info["vllm"] = None
    else:
        commit_info["vllm"] = None
        
    return commit_info


def get_run_command_reconstruction() -> Dict[str, Any]:
    """Reconstruct the original run.py command from environment variables.
    
    Analyzes environment to infer the command-line arguments that were used.
    
    Returns:
        Dict with reconstructed command and evidence sources
    """
    reconstructed_cmd = []
    evidence_sources = []
    
    # Model configuration
    model_repo = os.getenv("HF_MODEL_REPO_ID")
    if model_repo:
        model_name = model_repo.split("/")[-1] if "/" in model_repo else model_repo
        reconstructed_cmd.append(f"--model {model_name}")
        evidence_sources.append("HF_MODEL_REPO_ID")
    
    # Device configuration
    mesh_device = os.getenv("MESH_DEVICE")
    if mesh_device:
        reconstructed_cmd.append(f"--device {mesh_device.lower()}")
        evidence_sources.append("MESH_DEVICE")
    
    # Container detection
    container_indicators = [
        os.getenv("CONTAINER_APP_USERNAME"),
        os.getenv("HOSTNAME", "").startswith(("docker", "container")) or len(os.getenv("HOSTNAME", "")) == 12,
        os.path.exists("/.dockerenv"),
        os.getenv("SERVICE_PORT")
    ]
    
    if any(container_indicators):
        reconstructed_cmd.extend(["--workflow server", "--docker-server"])
        evidence_sources.extend(["container_detection", "SERVICE_PORT"])
    
    # Implementation
    model_impl = os.getenv("MODEL_IMPL")
    if model_impl and model_impl != "tt-transformers":
        reconstructed_cmd.append(f"--impl {model_impl}")
        evidence_sources.append("MODEL_IMPL")
    
    # Development mode
    if os.getenv("TT_METAL_ENV") == "dev":
        reconstructed_cmd.append("--dev-mode")
        evidence_sources.append("TT_METAL_ENV")
    
    # Service port
    service_port = os.getenv("SERVICE_PORT")
    if service_port and service_port != "8000":
        reconstructed_cmd.append(f"--service-port {service_port}")
        evidence_sources.append("SERVICE_PORT")
    
    # Configuration overrides
    if os.getenv("OVERRIDE_TT_CONFIG"):
        override_config = os.getenv("OVERRIDE_TT_CONFIG")
        reconstructed_cmd.append(f'--override-tt-config \'{override_config}\'')
        evidence_sources.append("OVERRIDE_TT_CONFIG")
    
    # VLLM overrides
    vllm_overrides = {}
    default_values = {"VLLM_MAX_MODEL_LEN": "131072", "VLLM_MAX_NUM_SEQS": "32"}
    
    for env_var, default_val in default_values.items():
        value = os.getenv(env_var)
        if value and value != default_val:
            key = env_var.replace("VLLM_", "").lower()
            vllm_overrides[key] = value
            evidence_sources.append(env_var)
    
    if vllm_overrides:
        override_str = json.dumps(vllm_overrides)
        reconstructed_cmd.append(f'--vllm-override-args \'{override_str}\'')
    
    return {
        "command": f"python3 run.py {' '.join(reconstructed_cmd)}" if reconstructed_cmd else None,
        "arguments": reconstructed_cmd,
        "evidence_sources": evidence_sources
    }


def _extract_container_id_from_cgroup() -> Optional[str]:
    """Extract container ID from /proc/self/cgroup file."""
    if not os.path.exists("/proc/self/cgroup"):
        return None
        
    try:
        with open("/proc/self/cgroup", "r") as f:
            cgroup_content = f.read()
            
        patterns = [
            r'/docker/([a-f0-9]{64})',
            r'docker-([a-f0-9]{64})',
            r'/([a-f0-9]{64})\.scope',
            r'containers\.slice/docker-([a-f0-9]{64})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cgroup_content)
            if match:
                return match.group(1)[:12]  # Return short ID
    except Exception as e:
        logger.debug(f"Error reading cgroup: {e}")
    
    return None


def _extract_container_id_from_mountinfo() -> Optional[str]:
    """Extract container ID from /proc/self/mountinfo file."""
    if not os.path.exists("/proc/self/mountinfo"):
        return None
        
    try:
        with open("/proc/self/mountinfo", "r") as f:
            mountinfo = f.read()
            
        if "docker" in mountinfo.lower():
            match = re.search(r'/var/lib/docker/containers/([a-f0-9]{64})', mountinfo)
            if match:
                return match.group(1)[:12]
    except Exception as e:
        logger.debug(f"Error reading mountinfo: {e}")
    
    return None


def _get_container_info_via_docker(container_id: str) -> Dict[str, Any]:
    """Get container information via docker inspect command."""
    info = {}
    
    if not shutil.which("docker"):
        return info
        
    try:
        result = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            inspect_data = json.loads(result.stdout)[0]
            config = inspect_data.get("Config", {})
            
            if config.get("Image"):
                info["image_name"] = config["Image"]
            if inspect_data.get("Image"):
                info["image_id"] = inspect_data["Image"]
            if inspect_data.get("Created"):
                info["created"] = inspect_data["Created"]
            
            info["extraction_method"] = "docker_inspect_success"
    except Exception as e:
        logger.debug(f"Could not inspect container: {e}")
    
    return info


def get_docker_container_info() -> Dict[str, Any]:
    """Extract Docker container information if running in a container.
    
    Returns:
        Dict with Docker container information (only includes available data)
    """
    # Check if running in a container
    container_indicators = [
        os.path.exists("/.dockerenv"),
        os.getenv("CONTAINER_APP_USERNAME"),
        os.getenv("HOSTNAME", "").startswith(("docker", "container")) or len(os.getenv("HOSTNAME", "")) == 12
    ]
    
    if not any(container_indicators):
        return {"in_container": False}
    
    docker_info = {"in_container": True}
    
    try:
        # Try various methods to extract container ID
        container_id = (
            _extract_container_id_from_cgroup() or
            _extract_container_id_from_mountinfo()
        )
        
        # Use hostname as fallback if it looks like a container ID
        if not container_id:
            hostname = os.getenv("HOSTNAME", "")
            if hostname and len(hostname) == 12 and re.match(r'^[a-f0-9]{12}$', hostname):
                container_id = hostname
        
        if container_id:
            docker_info["container_id"] = container_id
            docker_info["extraction_method"] = "cgroup"
            
            # Try to get additional info via docker inspect
            docker_inspect_info = _get_container_info_via_docker(container_id)
            docker_info.update(docker_inspect_info)
        
        # Set extraction method if not already set
        if "extraction_method" not in docker_info:
            if os.path.exists("/.dockerenv"):
                docker_info["extraction_method"] = "dockerenv_file"
            else:
                docker_info["extraction_method"] = "environment_indicators"
        
        # Look for image information in environment
        if "image_name" not in docker_info:
            for var in ["IMAGE_NAME", "DOCKER_IMAGE", "CONTAINER_IMAGE"]:
                value = os.getenv(var)
                if value:
                    docker_info["image_name"] = value
                    break
        
    except Exception as e:
        logger.warning(f"Error extracting Docker container info: {e}")
    
    return docker_info


def get_container_dependencies() -> Dict[str, Any]:
    """Get container dependency information if available.
        
    Returns:
        Dict with Python packages and extraction status
    """
    deps_info = {
        "python_packages": {},
        "python_packages_count": 0,
        "extraction_available": False
    }
    
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            packages = {}
            for line in result.stdout.strip().split('\n'):
                if line and '==' in line:
                    try:
                        name, version = line.split('==', 1)
                        packages[name] = version
                    except ValueError:
                        packages[line] = "unknown"
            
            deps_info.update({
                "python_packages": packages,
                "python_packages_count": len(packages),
                "extraction_available": True
            })
    except Exception as e:
        logger.debug(f"Could not get pip packages: {e}")
    
    return deps_info


def check_system_dependency(command: str) -> Dict[str, Any]:
    """Check if a system dependency is available and get its version.
    
    Args:
        command: Command to check (e.g., 'git', 'gcc', 'python3')
        
    Returns:
        Dict with availability, version, and path information
    """
    result = {
        "available": False,
        "version": None,
        "path": None,
        "error": None
    }
    
    try:
        path = shutil.which(command)
        if path:
            result["available"] = True
            result["path"] = path
            
            # Try common version commands
            for version_cmd in [[command, "--version"], [command, "-V"], [command, "version"]]:
                try:
                    version_result = subprocess.run(
                        version_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if version_result.returncode == 0:
                        result["version"] = version_result.stdout.strip().split('\n')[0]
                        break
                except Exception:
                    continue
                    
    except Exception as e:
        result["error"] = str(e)
        
    return result


def get_system_dependencies() -> Dict[str, Dict[str, Any]]:
    """Get status of critical system dependencies for CI reproduction.
    
    Returns:
        Dict mapping dependency names to their status information
    """
    critical_deps = [
        "gcc", "git", "python3", "curl", "wget", "cmake", "make", 
        "jq", "vim", "unzip", "zip", "rsync", "docker"
    ]
    
    return {dep: check_system_dependency(dep) for dep in critical_deps}


def generate_preconditions_json(output_path: Optional[Path] = None, 
                              include_sensitive: bool = False) -> Dict[str, Any]:
    """Generate CI-focused preconditions JSON for Docker container issue reproduction.
    
    Args:
        output_path: Optional path to save the JSON file
        include_sensitive: Whether to include sensitive environment variables
        
    Returns:
        Dictionary containing CI reproduction information
    """
    logger.info("Generating CI preconditions extraction...")
    
    # Gather all precondition data
    env_analysis = get_environment_analysis(include_sensitive)
    commit_info = get_commit_information()
    command_reconstruction = get_run_command_reconstruction()
    docker_info = get_docker_container_info()
    container_deps = get_container_dependencies()
    system_deps = get_system_dependencies()
    
    # Count sensitive variables
    filtered_env_vars = env_analysis["all_environment"]
    sensitive_count = sum(1 for value in filtered_env_vars.values() if value == "<REDACTED>")
    
    # Build environment variables section with comprehensive categorization
    environment_vars = {
        "statistics": env_analysis["statistics"],
        "sensitive_filtered_count": sensitive_count
    }
    
    for category_name, var_list in env_analysis["categories"].items():
        environment_vars[category_name] = {
            var: env_analysis["all_environment"].get(var) for var in var_list
        }
    
    preconditions = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment_vars": environment_vars,
        "commit_shas": commit_info,
        "run_command_reconstruction": command_reconstruction,
        "docker_container_info": docker_info,
        "system_dependencies": system_deps,
        "container_dependencies": container_deps
    }
    
    if output_path:
        logger.info(f"Saving CI preconditions to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(preconditions, f, indent=2)
        
    return preconditions


def main():
    """Main function to generate preconditions extraction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Extract CI preconditions for Docker container reproduction")
    parser.add_argument("--output", "-o", metavar="FILE", type=Path,
                       help="Output path for generated preconditions.json")
    parser.add_argument("--include-sensitive", action="store_true",
                       help="Include sensitive environment variables (tokens, keys, etc.)")
    
    args = parser.parse_args()
    
    try:
        preconditions = generate_preconditions_json(args.output, include_sensitive=args.include_sensitive)
        
        # Print summary
        logger.info("=== Complete Environment Extraction Summary ===")
        stats = preconditions['environment_vars']['statistics']
        
        logger.info(f"Total environment variables: {stats['total_variables']}")
        logger.info("Category breakdown:")
        for category, count in stats['category_breakdown'].items():
            logger.info(f"  - {category}: {count} variables")
        
        commit_shas_resolved = sum(1 for sha in preconditions['commit_shas'].values() if sha)
        logger.info(f"Commit SHAs resolved: {commit_shas_resolved}")
        logger.info(f"Run command extracted: {bool(preconditions['run_command_reconstruction'].get('command'))}")
        logger.info(f"In Docker container: {preconditions['docker_container_info']['in_container']}")
        
        sys_deps = preconditions['system_dependencies']
        available_deps = sum(1 for dep in sys_deps.values() if dep.get('available'))
        logger.info(f"System dependencies: {available_deps}/{len(sys_deps)} available")
        logger.info(f"Python packages extracted: {preconditions['container_dependencies']['python_packages_count']}")
        
    except Exception as e:
        logger.error(f"Failed to extract preconditions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 