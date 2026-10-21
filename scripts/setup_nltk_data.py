# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Install the pinned English NLTK data IFEval and IFBench scoring need.

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
# (archive path under packages/, archive sha256, {member path: sha256}, target dir).
# Punkt is what IFEval sentence scoring needs. Stopwords and the English
# perceptron tagger (JSON weights, no pickle) are what the IFBench verifier
# needs (count_stopwords, nltk.pos_tag); without them it tries NLTK's own
# downloader at import, which the runner proxy refuses.
PACKAGES = [
    (
        "tokenizers/punkt_tab.zip",
        "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
        {
            "punkt_tab/english/collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
            "punkt_tab/english/sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
            "punkt_tab/english/abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
            "punkt_tab/english/ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
        },
        "tokenizers",
    ),
    (
        "corpora/stopwords.zip",
        "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
        {
            "stopwords/english": "f6d005956f407dbc6ea32e5ff0c7e8e6f71488d3239b9023efdc7fc139d6375b",
        },
        "corpora",
    ),
    (
        "taggers/averaged_perceptron_tagger_eng.zip",
        "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
        {
            "averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json": "789e8e35f6fac656d9b7eccea29ba4e8a137cbb5b525dd31f39e4291a9e80832",
            "averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json": "d152ffa69d45b8357276341a3d592c6c025911d52ec0d9068d4f382560dd48b6",
            "averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json": "4c713731cb06727736962bc8cd94ad919217a040e76691059af99bc0c2c9c246",
        },
        "taggers",
    ),
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_package(data_root: Path, archive_path, archive_sha256, members, subdir) -> None:
    target = data_root / subdir
    if all(
        (target / name).is_file() and _sha256((target / name).read_bytes()) == digest
        for name, digest in members.items()
    ):
        return
    url = f"https://raw.githubusercontent.com/nltk/nltk_data/{REVISION}/packages/{archive_path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read(16 * 1024 * 1024)
    if _sha256(archive) != archive_sha256:
        raise ValueError(f"NLTK archive checksum mismatch: {archive_path}")
    # Read only named files. Never extract archive paths or pickle data.
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        contents = {name: source.read(name) for name in members}
    for name, data in contents.items():
        if _sha256(data) != members[name]:
            raise ValueError(f"NLTK file checksum mismatch: {name}")
    for name, data in contents.items():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(data)


def main() -> None:
    import nltk

    data_root = Path(sys.prefix) / "nltk_data"
    for package in PACKAGES:
        ensure_package(data_root, *package)
    # Exercise the same legacy lookup used by lm-eval 0.4.4 IFEval.
    tokenizer = nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    if tokenizer.tokenize("Hello world. Another sentence.") != [
        "Hello world.",
        "Another sentence.",
    ]:
        raise RuntimeError("NLTK sentence-tokenizer preflight failed")
    if "the" not in nltk.corpus.stopwords.words("english"):
        raise RuntimeError("NLTK stopwords preflight failed")
    if [t for _, t in nltk.pos_tag(["The", "dog", "runs", "."])][:2] != ["DT", "NN"]:
        raise RuntimeError("NLTK perceptron tagger preflight failed")
    print(f"NLTK English Punkt/stopwords/tagger ready: {data_root} (source {REVISION})")


if __name__ == "__main__":
    main()
