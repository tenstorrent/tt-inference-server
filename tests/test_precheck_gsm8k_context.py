import json
import sys

from scripts.release import precheck_gsm8k_context as precheck


def test_precheck_counts_batch_encoding_and_binds_selected_rows(
    monkeypatch, tmp_path
):
    rows = [
        {"question": "one", "answer": "1"},
        {"question": "two", "answer": "2"},
    ]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return {"input_ids": [[1] * (10 + len(messages[0]["content"]))]}

    monkeypatch.setattr(precheck, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        precheck.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: Tokenizer(),
    )
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precheck",
            "--tokenizer-root",
            str(tokenizer_root),
            "--output",
            str(output),
            "--samples",
            "2",
            "--max-output",
            "20",
            "--max-context",
            "100",
        ],
    )

    assert precheck.main() == 0
    receipt = json.loads(output.read_text())
    assert receipt["rendered_input_counts"] == [31, 31]
    assert receipt["max_total_tokens"] == 51
    assert receipt["fits"] is True
