# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Identify catalog entries that pin no docker image.

A catalog template (``workflows/model_specs/*.yaml``) "pins" its image when it
sets either ``version`` or ``docker_image``. With neither, ``ModelSpec`` falls
back to synthesizing a tag from the repo-wide ``VERSION`` file + commits (e.g.
``0.15.0-747215b-7678b70``), which is not a real published image. The helm
generator must skip those (model, device, engine, impl) entries.

This is read straight from the catalog YAML rather than from the loaded
``ModelSpec`` / ``ModelSpecTemplate`` objects on purpose: by the time a spec is
built, ``version`` has been defaulted to ``VERSION`` and ``docker_image`` has
been synthesized, so an *omitted* ``version`` is indistinguishable from one
explicitly set to the current ``VERSION`` (the catalog has both -- see
``audio_tts.yaml``). Only the raw YAML preserves key presence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

import yaml

from workflows.helm_generator.device import device_key
from workflows.workflow_types import DeviceTypes, InferenceEngine

# (model_name, device_key, engine_value, impl_id) -- the same identity tuple
# cli.assert_unique uses and the path a spec is written to in values.yaml.
SpecKey = Tuple[str, str, str, str]

CATALOG_DIR = Path(__file__).resolve().parents[2] / "workflows" / "model_specs"


def unpinned_image_spec_keys(catalog_dir: Path = CATALOG_DIR) -> Set[SpecKey]:
    """Return SpecKeys for catalog entries that set neither version nor
    docker_image. Globs every ``*.yaml`` in ``catalog_dir`` so new catalog
    categories are covered automatically.
    """
    keys: Set[SpecKey] = set()
    for path in sorted(catalog_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for tmpl in data.get("templates", []):
            if tmpl.get("version") is not None or tmpl.get("docker_image") is not None:
                continue
            engine = InferenceEngine[tmpl["inference_engine"]].value
            impl_id = tmpl["impl"]
            for weight in tmpl["weights"]:
                model_name = Path(weight).name
                for dms in tmpl["device_model_specs"]:
                    dev = device_key(DeviceTypes.from_string(dms["device"]))
                    keys.add((model_name, dev, engine, impl_id))
    return keys
