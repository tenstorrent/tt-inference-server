# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from io import StringIO

from ruamel.yaml import YAML

from workflows.helm_generator.base_mapper import COMMON_OWNED_PATHS
from workflows.helm_generator.merge import merge_spec, set_default_engine
from workflows.helm_generator.schema import (
    HelmEnvVar,
    HelmImage,
    HelmImplConfig,
    HelmProbe,
    HelmProbes,
    HelmResources,
    HelmModelSpec,
)
from workflows.helm_generator.yaml_io import dump_values, load_values


def _yaml():
    y = YAML(typ="rt")
    y.indent(mapping=2, sequence=4, offset=2)
    y.preserve_quotes = True
    y.width = 4096
    return y


def _load(text):
    return _yaml().load(StringIO(text))


def _dump(doc):
    buf = StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()


def _mapped(
    *,
    model="m",
    engine="vllm",
    device="galaxy",
    impl="impl_a",
    is_default=True,
    repository="ghcr.io/org/img",
    tag="1.0.0",
    progress=5400,
    initial_delay=2400,
    liveness_path=None,
    memory="20Gi",
    env=(),
) -> HelmModelSpec:
    cfg = HelmImplConfig(
        image=HelmImage(repository=repository, tag=tag),
        progress_deadline_seconds=progress,
        probes=HelmProbes(
            liveness=HelmProbe(initial_delay_seconds=initial_delay, path=liveness_path),
            readiness=HelmProbe(initial_delay_seconds=initial_delay),
        ),
        resources=HelmResources(requests_memory=memory),
        env=[HelmEnvVar(name=k, value=v) for k, v in env],
    )
    return HelmModelSpec(
        model_name=model,
        engine=engine,
        device_name=device,
        impl_id=impl,
        is_default=is_default,
        config=cfg,
    )


def test_inserts_new_model_with_full_tree():
    doc = _load("models: {}\n")
    result = merge_spec(doc, _mapped(), set(COMMON_OWNED_PATHS))

    assert result.changed and result.inserted_model
    entry = doc["models"]["m"]
    assert entry["vllm"]["galaxy"]["defaultImpl"] == "impl_a"
    assert entry["vllm"]["galaxy"]["impls"]["impl_a"]["image"]["tag"] == "1.0.0"


def test_inserts_new_engine_under_existing_model():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            image:\n"
        "              repository: r\n"
        "              tag: t\n"
    )
    result = merge_spec(
        doc, _mapped(engine="media", device="t3k", impl="impl_m"), set(COMMON_OWNED_PATHS)
    )

    assert result.changed and result.inserted_engine
    assert "media" in doc["models"]["m"]
    assert doc["models"]["m"]["vllm"]["galaxy"]["impls"]["impl_a"]["image"]["tag"] == "t"


def test_inserts_new_device_under_existing_engine():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      t3k:\n"
        "        defaultImpl: a\n"
        "        impls:\n"
        "          a: {}\n"
    )
    result = merge_spec(
        doc, _mapped(device="galaxy", impl="b"), set(COMMON_OWNED_PATHS)
    )

    assert result.changed and result.inserted_device
    assert doc["models"]["m"]["vllm"]["galaxy"]["defaultImpl"] == "b"


def test_inserts_new_impl_under_existing_device_without_changing_existing_default():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: a\n"
        "        impls:\n"
        "          a:\n"
        "            image:\n"
        "              repository: r\n"
        "              tag: t\n"
    )
    # New impl marked is_default=False -> existing defaultImpl unchanged
    result = merge_spec(
        doc, _mapped(impl="b", is_default=False), set(COMMON_OWNED_PATHS)
    )
    assert result.inserted_impl
    assert doc["models"]["m"]["vllm"]["galaxy"]["defaultImpl"] == "a"
    assert "b" in doc["models"]["m"]["vllm"]["galaxy"]["impls"]


def test_inserts_new_impl_marked_default_overrides_existing_default():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: a\n"
        "        impls:\n"
        "          a: {}\n"
    )
    result = merge_spec(
        doc, _mapped(impl="b", is_default=True), set(COMMON_OWNED_PATHS)
    )
    assert result.inserted_impl
    assert doc["models"]["m"]["vllm"]["galaxy"]["defaultImpl"] == "b"
    assert ("defaultImpl",) in result.updated_paths


def test_no_change_when_existing_matches_generated():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            progressDeadlineSeconds: 5400\n"
        "            image:\n"
        "              repository: ghcr.io/org/img\n"
        "              tag: '1.0.0'\n"
        "            resources:\n"
        "              requests:\n"
        "                memory: 20Gi\n"
        "            probes:\n"
        "              liveness:\n"
        "                initialDelaySeconds: 2400\n"
        "              readiness:\n"
        "                initialDelaySeconds: 2400\n"
    )
    before = _dump(doc)
    result = merge_spec(doc, _mapped(), set(COMMON_OWNED_PATHS))
    assert not result.changed
    assert _dump(doc) == before


def test_updates_only_differing_leaf_keys():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            progressDeadlineSeconds: 5400\n"
        "            image:\n"
        "              repository: ghcr.io/org/img\n"
        "              tag: '0.9.0'\n"
        "            resources:\n"
        "              requests:\n"
        "                memory: 20Gi\n"
        "            probes:\n"
        "              liveness:\n"
        "                initialDelaySeconds: 2400\n"
        "              readiness:\n"
        "                initialDelaySeconds: 2400\n"
    )
    result = merge_spec(doc, _mapped(tag="1.0.0"), set(COMMON_OWNED_PATHS))

    assert result.changed
    assert ("image", "tag") in result.updated_paths
    assert ("image", "repository") not in result.updated_paths
    assert (
        doc["models"]["m"]["vllm"]["galaxy"]["impls"]["impl_a"]["image"]["tag"]
        == "1.0.0"
    )


def test_preserves_inline_comment_through_leaf_update(tmp_path):
    src = tmp_path / "in.yaml"
    src.write_text(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            progressDeadlineSeconds: 5400\n"
        "            image:\n"
        "              repository: ghcr.io/org/img\n"
        "              tag: '0.9.0'  # pinned\n"
        "            resources:\n"
        "              requests:\n"
        "                memory: 20Gi\n"
        "            probes:\n"
        "              liveness:\n"
        "                initialDelaySeconds: 2400\n"
        "              readiness:\n"
        "                initialDelaySeconds: 2400\n"
    )
    doc = load_values(src)
    merge_spec(doc, _mapped(tag="1.0.0"), set(COMMON_OWNED_PATHS))
    out = tmp_path / "out.yaml"
    dump_values(doc, out)
    text = out.read_text()
    assert "1.0.0" in text
    assert "# pinned" in text


def test_preserves_user_added_keys_outside_owned_paths():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            progressDeadlineSeconds: 5400\n"
        "            image:\n"
        "              repository: ghcr.io/org/img\n"
        "              tag: '0.9.0'\n"
        "            resources:\n"
        "              requests:\n"
        "                memory: 20Gi\n"
        "              limits:\n"
        "                memory: 100Gi\n"
        "            probes:\n"
        "              liveness:\n"
        "                initialDelaySeconds: 2400\n"
        "              readiness:\n"
        "                initialDelaySeconds: 2400\n"
        "            nodeSelector:\n"
        "              kubernetes.io/hostname: galaxy-node-01\n"
    )
    merge_spec(doc, _mapped(tag="1.0.0"), set(COMMON_OWNED_PATHS))
    impl = doc["models"]["m"]["vllm"]["galaxy"]["impls"]["impl_a"]
    assert impl["resources"]["limits"]["memory"] == "100Gi"
    assert impl["nodeSelector"]["kubernetes.io/hostname"] == "galaxy-node-01"
    assert impl["image"]["tag"] == "1.0.0"


def test_env_list_replaced_when_different():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            progressDeadlineSeconds: 5400\n"
        "            image:\n"
        "              repository: ghcr.io/org/img\n"
        "              tag: '1.0.0'\n"
        "            resources:\n"
        "              requests:\n"
        "                memory: 20Gi\n"
        "            probes:\n"
        "              liveness:\n"
        "                initialDelaySeconds: 2400\n"
        "              readiness:\n"
        "                initialDelaySeconds: 2400\n"
        "            env:\n"
        "              - name: A\n"
        "                value: '1'\n"
    )
    result = merge_spec(
        doc, _mapped(env=[("A", "1"), ("B", "2")]), set(COMMON_OWNED_PATHS)
    )
    assert ("env",) in result.updated_paths
    env_list = doc["models"]["m"]["vllm"]["galaxy"]["impls"]["impl_a"]["env"]
    assert len(env_list) == 2
    assert env_list[1]["name"] == "B"


def test_env_compare_normalizes_int_vs_str():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a:\n"
        "            progressDeadlineSeconds: 5400\n"
        "            image:\n"
        "              repository: ghcr.io/org/img\n"
        "              tag: '1.0.0'\n"
        "            resources:\n"
        "              requests:\n"
        "                memory: 20Gi\n"
        "            probes:\n"
        "              liveness:\n"
        "                initialDelaySeconds: 2400\n"
        "              readiness:\n"
        "                initialDelaySeconds: 2400\n"
        "            env:\n"
        "              - name: VLLM_RPC_TIMEOUT\n"
        "                value: 900000\n"
    )
    result = merge_spec(
        doc,
        _mapped(env=[("VLLM_RPC_TIMEOUT", "900000")]),
        set(COMMON_OWNED_PATHS),
    )
    assert not result.changed


def test_set_default_engine_writes_and_is_idempotent():
    doc = _load(
        "models:\n"
        "  m:\n"
        "    vllm:\n"
        "      galaxy:\n"
        "        defaultImpl: impl_a\n"
        "        impls:\n"
        "          impl_a: {}\n"
    )
    assert set_default_engine(doc, "m", "vllm")
    assert doc["models"]["m"]["defaultEngine"] == "vllm"
    assert not set_default_engine(doc, "m", "vllm")
