import pytest

from scripts.check_poc_environment import check_environment


def test_matching_endpoint_and_credentials_are_allowed(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("HF_TOKEN=private\nOPENAI_BASE_URL=http://aig:30000/v1\n")
    receipt = check_environment("http://aig:30000", str(tmp_path), {}, dotenv)
    assert receipt["server_url"] == "http://aig:30000"
    assert "private" not in str(receipt)


@pytest.mark.parametrize("source", ["environment", ".env"])
@pytest.mark.parametrize(
    "key",
    [
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "MODEL_SPECS_ENV",
        "OVERRIDE_BENCHMARK_TARGETS",
        "HARBOR_ENV_TYPE",
        "HARBOR_TIMEOUT_SEC",
        "HARBOR_ENFORCE_AGENT_DEADLINE",
        "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES",
        "CACHE_ROOT",
    ],
)
def test_conflicts_fail_without_printing_values(tmp_path, source, key):
    dotenv = tmp_path / ".env"
    env = {}
    if source == ".env":
        dotenv.write_text(f"{key}=private-wrong-value\n")
    else:
        env[key] = "private-wrong-value"
    with pytest.raises(ValueError) as error:
        check_environment("http://aig:30000", str(tmp_path), env, dotenv)
    assert f"{source}:{key}" in str(error.value)
    assert "private-wrong-value" not in str(error.value)


def test_origin_cannot_contain_credentials(tmp_path):
    with pytest.raises(ValueError, match="origin"):
        check_environment(
            "http://user:secret@aig", str(tmp_path), {}, tmp_path / ".env"
        )
