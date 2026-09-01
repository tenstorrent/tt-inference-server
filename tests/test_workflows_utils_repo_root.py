import pytest

from workflows.utils import get_repo_root_path


def test_repo_root_override_supports_source_archive(monkeypatch, tmp_path):
    (tmp_path / "VERSION").write_text("0.0.0\n")
    (tmp_path / "workflows").mkdir()
    (tmp_path / "llm_module").mkdir()
    monkeypatch.setenv("TTIS_REPO_ROOT", str(tmp_path))

    assert get_repo_root_path() == tmp_path.resolve()


@pytest.mark.parametrize("value", ["relative", "/missing/ttis-source-root"])
def test_repo_root_override_fails_closed(monkeypatch, value):
    monkeypatch.setenv("TTIS_REPO_ROOT", value)

    with pytest.raises((ValueError, FileNotFoundError), match="TTIS_REPO_ROOT"):
        get_repo_root_path()
