# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from pathlib import Path
from datetime import datetime

# get current year
current_year = datetime.now().year


# * SPDX header content
SPDX_HEADER = """# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © """

SPDX_DATE = str(current_year) + " Tenstorrent USA, Inc.\n"


def add_spdx_header(file_path):
    with open(file_path, "r+") as file:
        content = file.read()
        if "SPDX-License-Identifier" not in content:
            file.seek(0, 0)
            file.write(SPDX_HEADER + SPDX_DATE + "\n" + content)


if __name__ == "__main__":
    # List of directories to process here
    repo_root = Path(__file__).resolve().parent.parent
    directories_to_process = [
        repo_root / "vllm-tt-metal",
        repo_root / "tt-media-server",
        repo_root / "tt-sglang-plugin",
        repo_root / "tt-vllm-plugin",
        repo_root / "dynamo_frontend",
        repo_root / "launchers",
        repo_root / "llm_module",
        repo_root / "report_module",
        repo_root / "test_module",
        repo_root / "workflow_module",
        repo_root / "workflows",
        repo_root / "mooncake",
        repo_root / "scripts",
        repo_root / "tests",
        repo_root / "test_fixtures",
        repo_root / "utils",
        repo_root / "reference_config",
    ]
    # Walk through the specified directories and add the header to all relevant files
    for directory in directories_to_process:
        for file_path in directory.rglob("*"):
            # Check if the file is Python, Dockerfile, or Bash
            if file_path.suffix in (".py", ".sh") or file_path.name.endswith(
                "Dockerfile"
            ):
                add_spdx_header(file_path)
