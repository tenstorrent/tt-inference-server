# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import hashlib
import io
import zipfile

import pytest

from scripts import setup_nltk_data


def test_untrusted_download_does_not_write_tokenizer_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        setup_nltk_data.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        setup_nltk_data.ensure_punkt_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_verified_data_repairs_partial_cache_and_reuses_it(tmp_path, monkeypatch):
    files = {"collocations.tab": b"test data", "sent_starters.txt": b"other data"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(f"punkt_tab/english/{name}", data)
        archive.writestr("../../outside", b"must not be extracted")
    payload = buffer.getvalue()
    monkeypatch.setattr(
        setup_nltk_data, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(
        setup_nltk_data,
        "ENGLISH_FILES",
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    )
    monkeypatch.setattr(
        setup_nltk_data.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(payload),
    )
    target = tmp_path / "tokenizers" / "punkt_tab" / "english"
    target.mkdir(parents=True)
    (target / "collocations.tab").write_bytes(b"partial")
    setup_nltk_data.ensure_punkt_data(tmp_path)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == files
    assert not (tmp_path / "outside").exists()

    def no_download(*args, **kwargs):
        pytest.fail("Verified cache should not need network access")

    monkeypatch.setattr(setup_nltk_data.urllib.request, "urlopen", no_download)
    setup_nltk_data.ensure_punkt_data(tmp_path)
