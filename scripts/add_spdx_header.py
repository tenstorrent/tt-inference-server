# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

from pathlib import Path


# * SPDX header content
SPDX_HEADER = """# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
"""


def add_spdx_header(file_path):
    with open(file_path, "r+") as file:
        content = file.read()
        if "SPDX-License-Identifier" not in content:
            file.seek(0, 0)
            file.write(SPDX_HEADER + "\n" + content)


if __name__ == "__main__":
    # List of directories to process here
    repo_root = Path(__file__).resolve().parent.parent
    directories_to_process = [
        repo_root / "tt-metal-llama3-70b",
        repo_root / "tt-metal-mistral-7b",
    ]
    # Walk through the specified directories and add the header to all relevant files
    for directory in directories_to_process:
        for file_path in directory.rglob("*"):
            # Check if the file is Python, Dockerfile, or Bash
            if file_path.suffix in (".py", ".sh") or file_path.name.endswith(
                "Dockerfile"
            ):
                add_spdx_header(file_path)