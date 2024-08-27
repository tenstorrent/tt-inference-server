#* List of directories to process here ; in a relative path to this script
directories_to_process = ["../tt-metal-llama3-70b", "../tt-metal-mistral-7b"]

#* SPDX header content
SPDX_HEADER = """# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
"""

import os

def add_spdx_header(file_path):
    with open(file_path, 'r+') as file:
        content = file.read()
        if "SPDX-License-Identifier" not in content:
            file.seek(0, 0)
            file.write(SPDX_HEADER + "\n" + content)


# Walk through the specified directories and add the header to all relevant files
for directory in directories_to_process:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", "Dockerfile", ".sh")):  # Check if the file is Python, Dockerfile, or Bash
                file_path = os.path.join(root, file)  # Construct the file path
                add_spdx_header(file_path) 
