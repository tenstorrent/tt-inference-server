# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from scripts.release.promote_dev_spec_to_prod import (
    iter_implementations,
    model_name_from_weight,
)


def test_model_name_from_weight_strips_org_prefix():
    assert model_name_from_weight("meta-llama/Llama-3.1-8B-Instruct") == (
        "Llama-3.1-8B-Instruct"
    )
    assert model_name_from_weight("openai/gpt-oss-20b") == "gpt-oss-20b"


def test_iter_implementations_flat_shape():
    entry = {"inference_engine": "FORGE", "ci": {"nightly": {"devices": ["P150"]}}}
    assert list(iter_implementations(entry)) == [entry]


def test_iter_implementations_array_shape():
    impl_a = {"inference_engine": "vLLM", "ci": {}}
    impl_b = {"inference_engine": "FORGE", "ci": {}}
    entry = {"implementations": [impl_a, impl_b]}
    assert list(iter_implementations(entry)) == [impl_a, impl_b]
