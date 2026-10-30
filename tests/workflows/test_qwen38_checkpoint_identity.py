# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import hashlib
import json
from workflows.qwen38_checkpoint_identity import verify


def test_rejects_changed_shard_and_retains_failure_receipt(tmp_path):
    weights = tmp_path / 'weights'; weights.mkdir()
    (weights / 'model-00001-of-00001.safetensors').write_bytes(b'changed')
    expected = {'model': 'test', 'revision': 'a' * 40, 'files': {'model-00001-of-00001.safetensors': {'sha256': hashlib.sha256(b'original').hexdigest(), 'bytes': 8}}}
    receipt = verify(weights, expected)
    assert receipt['ok'] is False
    assert receipt['files'][0]['matches'] is False


def test_rejects_missing_and_extra_shards(tmp_path):
    weights = tmp_path / 'weights'; weights.mkdir()
    (weights / 'unlisted.safetensors').write_bytes(b'x')
    expected = {'model': 'test', 'revision': 'a' * 40, 'files': {'missing.safetensors': {'sha256': '0' * 64}}}
    receipt = verify(weights, expected)
    assert receipt['ok'] is False
    assert receipt['unexpected_shards'] == ['unlisted.safetensors']
    assert receipt['files'][0]['error'] == 'missing'


def test_verified_symlinked_snapshot(tmp_path):
    blob = tmp_path / 'blob'; blob.write_bytes(b'weights')
    weights = tmp_path / 'weights'; weights.mkdir()
    (weights / 'model.safetensors').symlink_to(blob)
    expected = {'model': 'test', 'revision': 'a' * 40, 'files': {'model.safetensors': {'sha256': hashlib.sha256(b'weights').hexdigest(), 'bytes': 7}}}
    receipt = verify(weights, expected)
    assert receipt['ok'] is True
    assert receipt['files'][0]['sha256'] == expected['files']['model.safetensors']['sha256']
