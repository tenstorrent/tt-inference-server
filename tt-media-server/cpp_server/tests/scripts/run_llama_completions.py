# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Run the Llama HTTP contract tests with an isolated, synthetic tokenizer.

The worker is mocked. These tests check routing, validation and token transport,
not model tokenization quality. No gated tokenizer download is needed.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="llama-completions-", dir=executable.parent) as directory:
        root = Path(directory)
        binary = root / "bin" / executable.name
        binary.parent.mkdir()
        # /proc/self/exe must point into this tree: a symlink would resolve back
        # to the original build and select its production tokenizer directory.
        shutil.copy2(executable, binary)
        assets = root / "tokenizers" / "meta-llama" / "Llama-3.2-1B-Instruct"
        assets.mkdir(parents=True)
        tokenizer = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {"token": 0, "[UNK]": 1},
                "unk_token": "[UNK]",
            },
        }
        (assets / "tokenizer.json").write_text(json.dumps(tokenizer))
        env = dict(os.environ, MODEL="meta-llama/Llama-3.2-1B-Instruct")
        # Preserve the original binary's colocated shared-library lookup.
        env["LD_LIBRARY_PATH"] = str(executable.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        return subprocess.run(
            [str(binary), "--gtest_filter=MainIntegrationTest.Completions*"],
            env=env,
            check=False,
        ).returncode


if __name__ == "__main__":
    sys.exit(main())
