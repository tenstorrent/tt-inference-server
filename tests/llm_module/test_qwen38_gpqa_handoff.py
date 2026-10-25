# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import json
import random
from pathlib import Path

import pytest

from llm_module.qwen38_gpqa_handoff import digest, process_docs


class Dataset(list):
    def map(self, fn, *, load_from_cache_file):
        assert load_from_cache_file is False
        return Dataset([{**doc, **fn(doc)} for doc in self])


def test_frozen_choices_ignore_global_seed_and_keep_bracketed_text(tmp_path):
    raw = Dataset([{"Question": "q", "Incorrect Answer 1": "[A] x", "Incorrect Answer 2": "y",
                    "Incorrect Answer 3": "z", "Correct Answer": "w"}])
    choices = ["[A] x", "y", "z", "w"]
    random.Random(0).shuffle(choices)
    expected = [{**raw[0], **{f"choice{i+1}": x for i, x in enumerate(choices)},
                 "choices": choices, "answer": f"({chr(65+choices.index('w'))})"}]
    h = {"population": 1, "population_sha256": digest(expected), "doc_ids": [0],
         "document_sha256": [digest(expected[0])], "manifest_sha256": "test"}
    random.seed(1234)
    before = random.getstate()
    assert process_docs(raw, h, tmp_path / "identity.json") == expected
    assert random.getstate() == before
    assert json.loads((tmp_path / "identity.json").read_text())["status"] == "pass"
    raw[0]["Question"] = "different population"
    with pytest.raises(ValueError, match="identity failed before scoring"):
        process_docs(raw, h, tmp_path / "bad.json")
    assert json.loads((tmp_path / "bad.json").read_text())["status"] == "fail"
