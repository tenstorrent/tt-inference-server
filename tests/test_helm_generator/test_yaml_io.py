# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path

from workflows.helm_generator.yaml_io import dump_values, dumps_values, load_values

FIXTURES = Path(__file__).parent / "fixtures"


def test_roundtrip_preserves_top_level_comments(tmp_path):
    src = FIXTURES / "test_values.yaml"
    doc = load_values(src)
    out = tmp_path / "values.yaml"
    dump_values(doc, out)
    rewritten = out.read_text()
    assert "Required at install time" in rewritten
    assert "Defaults" in rewritten
    assert "Per-model configuration" in rewritten


def test_load_returns_mutable_mapping():
    src = FIXTURES / "test_values.yaml"
    doc = load_values(src)
    assert "models" in doc
    entry = doc["models"]["Llama-3.1-8B-Instruct"]
    assert entry["defaultEngine"] == "vllm"
    assert (
        entry["vllm"]["galaxy"]["impls"][entry["vllm"]["galaxy"]["defaultImpl"]][
            "image"
        ]["tag"]
    )


def test_dumps_emits_yaml_string():
    doc = load_values(FIXTURES / "test_values.yaml")
    text = dumps_values(doc)
    assert text.startswith("#") or text.startswith("model")
    assert "Llama-3.1-8B-Instruct" in text


def test_setitem_preserves_inline_comment_on_existing_key(tmp_path):
    src = tmp_path / "src.yaml"
    src.write_text(
        "models:\n"
        "  foo:\n"
        "    image:\n"
        '      tag: "0.10.0"  # pinned\n'
    )
    doc = load_values(src)
    doc["models"]["foo"]["image"]["tag"] = "0.11.0"
    out = tmp_path / "out.yaml"
    dump_values(doc, out)
    assert "# pinned" in out.read_text()
    assert "0.11.0" in out.read_text()
