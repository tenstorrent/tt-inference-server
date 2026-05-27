# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.helm_generator.media.mapper import MediaMapper


def test_media_engine_and_liveness_path(media_spec):
    mapped = MediaMapper().map(media_spec)
    assert mapped.engine == "media"
    assert mapped.model_name == "whisper-large-v3"
    cfg = mapped.config
    assert cfg.probes.liveness.path == "/tt-liveness"
    assert cfg.probes.readiness.path is None


def test_media_image_split(media_spec):
    cfg = MediaMapper().map(media_spec).config
    assert cfg.image.repository == "ghcr.io/acme/media-server"
    assert cfg.image.tag == "0.11.1-bac8b34"


def test_media_resources_from_explicit_min_ram(media_spec):
    cfg = MediaMapper().map(media_spec).config
    assert cfg.resources.requests_memory == "6Gi"


def test_media_owned_paths_include_liveness_path():
    paths = MediaMapper().owned_leaf_paths()
    assert ("probes", "liveness", "path") in paths
