# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Preserve stage-11 GPQA documents while keeping the r1 prompt/scorer."""
import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def process_docs(dataset, handoff, evidence_path):
    # This is the exact stage-11 cot_zeroshot normalization. The upstream r1
    # processor additionally removes bracketed text, so merely resetting its
    # RNG would not preserve the frozen choice strings.
    def preprocess(text):
        if text is None:
            return " "
        return text.strip().replace(" [title]", ". ").replace("  ", " ")

    rng = random.Random(0)

    def process(doc):
        choices = [preprocess(doc[k]) for k in (
            "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer")]
        rng.shuffle(choices)
        correct = choices.index(preprocess(doc["Correct Answer"]))
        return {**{f"choice{i + 1}": value for i, value in enumerate(choices)},
                "choices": choices, "answer": f"({chr(65 + correct)})"}

    processed = dataset.map(process, load_from_cache_file=False)
    docs = list(processed)
    actual = [digest(docs[i]) for i in handoff["doc_ids"]]
    population_hash = digest(docs)
    checks = {
        "population_count": len(docs) == handoff["population"],
        "population_sha256": population_hash == handoff["population_sha256"],
        "frozen_documents": actual == handoff["document_sha256"],
    }
    evidence = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
                "population": len(docs), "population_sha256": population_hash,
                "doc_ids": handoff["doc_ids"], "document_sha256": actual,
                "processor_seed": 0, "map_cache": "recompute",
                "source_manifest_sha256": handoff["manifest_sha256"]}
    Path(evidence_path).write_text(json.dumps(evidence, indent=2) + "\n")
    if not all(checks.values()):
        raise ValueError(f"Goal-12 GPQA handoff identity failed before scoring: {checks}")
    return processed


def install(handoff_path, evidence_path):
    """Replace only the selected YAML processor before TaskManager loads it."""
    import lm_eval.utils
    handoff = json.loads(Path(handoff_path).read_text())["gpqa_handoff"]
    original = lm_eval.utils.import_function

    def import_function(loader, node, yaml_path):
        path = Path(yaml_path)
        if (path.name == "gpqa_reasoning_diamond.yaml" and path.parent.name == "r1_evals"
                and loader.construct_scalar(node) == "utils.process_gpqa_docs"):
            return lambda dataset: process_docs(dataset, handoff, evidence_path)
        return original(loader, node, yaml_path)

    lm_eval.utils.import_function = import_function
