from pathlib import Path


def test_swebench_verifier_pin_uses_valid_container_names():
    requirements = (
        Path(__file__).parents[1] / "requirements" / "evals-agentic.txt"
    ).read_text(encoding="utf-8")

    assert "swebench==5.0.2" in requirements
    assert "bc2a82af2874e26fa6f206ae0ad9017c4768daa2" not in requirements


def test_agentic_tokenizer_supports_gemma4_checkpoint_contract():
    requirements = (
        Path(__file__).parents[1] / "requirements" / "evals-agentic.txt"
    ).read_text(encoding="utf-8")

    assert "transformers==5.15.0" in requirements
    assert "transformers==4.57.1" not in requirements
