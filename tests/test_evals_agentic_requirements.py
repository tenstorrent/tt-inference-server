from pathlib import Path


def test_swebench_verifier_pin_uses_valid_container_names():
    requirements = (
        Path(__file__).parents[1] / "requirements" / "evals-agentic.txt"
    ).read_text(encoding="utf-8")

    assert "swebench==5.0.2" in requirements
    assert "bc2a82af2874e26fa6f206ae0ad9017c4768daa2" not in requirements


def test_agentic_driver_has_its_yaml_runtime_dependency():
    requirements = (
        Path(__file__).parents[1] / "requirements" / "evals-agentic.txt"
    ).read_text(encoding="utf-8")

    assert "pyyaml==6.0.3" in requirements
