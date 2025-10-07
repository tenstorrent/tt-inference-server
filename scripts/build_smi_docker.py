# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys
import subprocess
from pathlib import Path
import argparse

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def get_repo_root_path() -> Path:
    """Gets the root of the repository (assumes this script is in a 'scripts' dir)."""
    return Path(__file__).resolve().parent.parent


def run_command_with_logging(command, check=True, cwd=None):
    """
    Run a command and stream its output to the logger in real-time.
    """
    if isinstance(command, str):
        command = command.split()

    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd,
        )

        for line in iter(process.stdout.readline, ""):
            if line.strip():
                logger.info(line.strip())

        process.wait()

        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        return process.returncode
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Command failed: {e}")
        if check:
            raise
        return 1


def check_image_exists_remote(image_tag):
    """
    Check if a Docker image exists in the remote registry.
    """
    logger.info(f"Checking for remote image: {image_tag}")
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image_tag],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking remote image {image_tag}: {e}")
        return False


def check_image_exists_local(image_tag):
    """
    Check if a Docker image exists locally.
    """
    logger.info(f"Checking for local image: {image_tag}")
    try:
        result = subprocess.run(
            ["docker", "inspect", "--type=image", image_tag],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking local image {image_tag}: {e}")
        return False


def get_image_tag(tt_smi_version, ubuntu_version,
                  image_base_name="ghcr.io/tenstorrent/tt-inference-server/tt-smi"):
    """
    Generate the full Docker image name and tag, with OS info in the name.
    Example: ghcr.io/tenstorrent/tt-smi-ubuntu-22.04-amd64:1.2.3
    """
    # Use the provided version or default to "latest" if none is given
    version_tag = tt_smi_version if tt_smi_version else "latest"
    os_arch_tag = f"ubuntu-{ubuntu_version}-amd64"

    # Combine the base name and OS/arch for the full image name
    full_image_name = f"{image_base_name}-{os_arch_tag}"

    # Combine the full name and version for the final tag
    return f"{full_image_name}:{version_tag}"


def build_smi_image(image_tag, tt_smi_version, logger):
    """
    Build the tt-smi Docker image, passing the version as a build argument.
    The Dockerfile path is hard-coded.
    """
    repo_root = get_repo_root_path()
    dockerfile_path = repo_root / "deployment/tt-smi.Dockerfile"

    logger.info(f"Building tt-smi image: {image_tag}")
    logger.info(f"Using Dockerfile: {dockerfile_path.relative_to(repo_root)}")

    build_command = [
        "docker",
        "build",
        "-t",
        image_tag,
        "-f",
        str(dockerfile_path),
    ]

    # Conditionally add the build argument if a version is specified
    if tt_smi_version:
        build_command.extend(["--build-arg", f"TT_SMI_VERSION={tt_smi_version}"])
    
    # Add the build context at the end
    build_command.append(str(repo_root))

    run_command_with_logging(build_command, check=True, cwd=repo_root)
    logger.info(f"Successfully built image: {image_tag}")


def push_image(image_tag, logger):
    """
    Push a Docker image to the registry.
    """
    logger.info(f"Pushing image: {image_tag}")
    push_command = ["docker", "push", image_tag]
    run_command_with_logging(push_command, check=True)
    logger.info(f"Successfully pushed image: {image_tag}")


def main():
    """Main function to orchestrate the build process."""
    parser = argparse.ArgumentParser(description="Build Docker image for tt-smi.")
    parser.add_argument(
        "--build-tt-smi-version",
        type=str,
        default="",
        help="Version of tt-smi to install (e.g., '1.2.3'). If not provided, installs the latest version."
    )
    parser.add_argument(
        "--force-build", action="store_true", help="Force rebuild even if image exists."
    )
    parser.add_argument(
        "--push", action="store_true", help="Push the container to the registry after building."
    )
    parser.add_argument(
        "--ubuntu-version",
        type=str,
        default="22.04",
        help="Ubuntu version to use for the base image.",
        choices={"22.04"},
    )
    args = parser.parse_args()

    logger.info("--- Starting tt-smi Docker Build ---")
    logger.info(f"tt-smi Version: {args.build_tt_smi_version or 'latest'}")
    logger.info(f"Force Build: {args.force_build}")
    logger.info(f"Push to Registry: {args.push}")
    logger.info(f"Ubuntu Version: {args.ubuntu_version}")

    try:
        # 1. Generate the image tag
        image_tag = get_image_tag(args.build_tt_smi_version, args.ubuntu_version)
        logger.info(f"Generated image tag: {image_tag}")

        # 2. Check if the image needs to be built
        should_build = True
        if not args.force_build:
            if check_image_exists_local(image_tag) or check_image_exists_remote(image_tag):
                logger.info("Image already exists locally or remotely. Skipping build.")
                logger.info("Use --force-build to override.")
                should_build = False

        # 3. Build the image if necessary
        if should_build:
            build_smi_image(image_tag, args.build_tt_smi_version, logger)
        
        # 4. Push the image if requested and it exists locally
        if args.push:
            if not check_image_exists_local(image_tag):
                logger.error(f"Cannot push image {image_tag} as it was not found locally.")
                sys.exit(1)

            push_image(image_tag, logger)

        logger.info("--- Build process completed successfully! ---")

    except FileNotFoundError as e:
        logger.error(f"Error: A required file was not found. {e}")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.error("Error: A command failed to execute. See logs above for details.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
