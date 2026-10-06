# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Install the pinned English Punkt data required by IFEval sentence scoring.

NLTK's general downloader rejects proxied requests. Fetch only this immutable,
hash-verified upstream archive; keep NLTK's network protections enabled.
Run with the evaluation venv's Python so its standard data path is used.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
ARCHIVE_URL = (
    f"https://raw.githubusercontent.com/nltk/nltk_data/{REVISION}"
    "/packages/tokenizers/punkt_tab.zip"
)
ARCHIVE_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
ENGLISH_FILES = {
    "collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
    "abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
}


def ensure_punkt_data(data_root: Path) -> None:
    target = data_root / "tokenizers" / "punkt_tab" / "english"
    if all(
        (target / name).is_file()
        and hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
        for name, digest in ENGLISH_FILES.items()
    ):
        return
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
        archive = response.read(8 * 1024 * 1024)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("NLTK Punkt archive checksum mismatch")
    # Read only named text files. Never extract archive paths or pickle data.
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        contents = {
            name: source.read(f"punkt_tab/english/{name}") for name in ENGLISH_FILES
        }
    for name, data in contents.items():
        if hashlib.sha256(data).hexdigest() != ENGLISH_FILES[name]:
            raise ValueError(f"NLTK Punkt file checksum mismatch: {name}")
    target.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (target / name).write_bytes(data)


def main() -> None:
    import nltk

    data_root = Path(sys.prefix) / "nltk_data"
    ensure_punkt_data(data_root)
    # Exercise the same legacy lookup used by lm-eval 0.4.4 IFEval.
    tokenizer = nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    if tokenizer.tokenize("Hello world. Another sentence.") != [
        "Hello world.",
        "Another sentence.",
    ]:
        raise RuntimeError("NLTK sentence-tokenizer preflight failed")
    print(f"NLTK English Punkt ready: {data_root} (source {REVISION})")


if __name__ == "__main__":
    main()
