#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the env-driven Harbor environment selection in agentic evals.

TerminalBenchEvalConfig picks its Harbor environment (and the cluster knobs that
environment needs) from the process environment, so the same eval config runs
trials as local containers on a laptop and as pods on a cluster in CI without a
per-model edit. The defaults are resolved per instantiation via
``dataclasses.field(default_factory=...)``, which is what lets a workflow set
the variables before importing the config.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reference_config.evals.eval_config import TerminalBenchEvalConfig


def _config(**overrides) -> TerminalBenchEvalConfig:
    base = dict(dataset="terminal-bench/terminal-bench-2", agent="terminus-2")
    base.update(overrides)
    return TerminalBenchEvalConfig(**base)


def test_defaults_to_docker_with_no_env(monkeypatch):
    for var in ("HARBOR_ENV_TYPE", "HARBOR_K8S_NAMESPACE", "HARBOR_TIMEOUT_SEC"):
        monkeypatch.delenv(var, raising=False)

    cfg = _config()

    assert cfg.environment_type == "docker"
    # No kwargs on the docker path: harbor's docker environment rejects the
    # kubernetes-only keys, so they must not leak into a local run.
    assert cfg.environment_kwargs == {}
    assert cfg.harbor_timeout_sec is None


def test_kubernetes_env_type_emits_namespace_and_image_mode(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    for var in (
        "HARBOR_K8S_NAMESPACE",
        "HARBOR_K8S_IMAGE_MODE",
        "HARBOR_K8S_KUBECONFIG",
        "HARBOR_K8S_NODE_SELECTOR",
        "HARBOR_K8S_POD_LABELS",
        "HARBOR_K8S_REGISTRY_INSECURE",
        "HARBOR_K8S_SKIP_IMAGE_CHECK",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = _config()

    assert cfg.environment_type == "kubernetes"
    assert cfg.environment_kwargs == {"namespace": "default", "image_mode": "prebuilt"}


def test_in_cluster_run_omits_kubeconfig(monkeypatch):
    """An ARC runner pod sets no kubeconfig; harbor then loads the SA."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("HARBOR_K8S_NAMESPACE", "harbor-kube-env")
    monkeypatch.setenv("HARBOR_K8S_NODE_SELECTOR", '{"tt-pool": "shield"}')
    monkeypatch.delenv("HARBOR_K8S_KUBECONFIG", raising=False)
    monkeypatch.delenv("HARBOR_K8S_CONTEXT", raising=False)

    cfg = _config()

    assert cfg.environment_kwargs == {
        "namespace": "harbor-kube-env",
        "image_mode": "prebuilt",
        "node_selector": {"tt-pool": "shield"},
    }
    # Absent keys, not None values: harbor's KubernetesEnvironment treats a
    # present-but-None kubeconfig differently from an omitted one.
    assert "kubeconfig" not in cfg.environment_kwargs
    assert "context" not in cfg.environment_kwargs


def test_passthrough_vars_reach_kwargs(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("HARBOR_K8S_NAMESPACE", "harbor-kube-env")
    monkeypatch.setenv("HARBOR_K8S_IMAGE_MODE", "build-and-push")
    monkeypatch.setenv("HARBOR_K8S_CONTEXT", "kix1")
    monkeypatch.setenv("HARBOR_K8S_IMAGE_REGISTRY", "registry.local/harbor")
    monkeypatch.setenv("HARBOR_K8S_IMAGE_PULL_SECRET", "regcred")
    monkeypatch.setenv("HARBOR_K8S_SERVICE_ACCOUNT", "harbor-task")

    cfg = _config()

    assert cfg.environment_kwargs == {
        "namespace": "harbor-kube-env",
        "image_mode": "build-and-push",
        "context": "kix1",
        "image_registry": "registry.local/harbor",
        "image_pull_secret": "regcred",
        "service_account": "harbor-task",
    }


def test_registry_boolean_flags_reach_kubernetes_kwargs(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("HARBOR_K8S_REGISTRY_INSECURE", "true")
    monkeypatch.setenv("HARBOR_K8S_SKIP_IMAGE_CHECK", "false")

    cfg = _config()

    assert cfg.environment_kwargs["registry_insecure"] is True
    assert cfg.environment_kwargs["skip_image_check"] is False


def test_registry_boolean_flags_reject_invalid_values(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("HARBOR_K8S_REGISTRY_INSECURE", "sometimes")

    with pytest.raises(ValueError, match="HARBOR_K8S_REGISTRY_INSECURE"):
        _config()


def test_pod_labels_reach_kubernetes_kwargs(monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv(
        "HARBOR_K8S_POD_LABELS",
        '{"ci-run-id": "123456789", "ci-workflow": "agentic"}',
    )

    cfg = _config()

    assert cfg.environment_kwargs["pod_labels"] == {
        "ci-run-id": "123456789",
        "ci-workflow": "agentic",
    }


@pytest.mark.parametrize(
    "labels",
    [
        '["not", "a", "mapping"]',
        '{"ci-run-id": 123456789}',
    ],
)
def test_pod_labels_must_be_a_json_string_map(monkeypatch, labels):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("HARBOR_K8S_POD_LABELS", labels)

    with pytest.raises(ValueError, match="JSON string map"):
        _config()


def test_kubeconfig_path_is_never_a_kwarg(monkeypatch):
    """``KUBECONFIG`` is Harbor's own lookup, not something we forward.

    ``KubernetesEnvironment.__init__`` takes namespace/context/image_mode/
    image_registry/skip_image_check and nothing else; the file itself comes from
    ``KUBECONFIG`` -> ``~/.kube/config`` -> the in-cluster service account
    inside ``BaseKubernetesEnvironment.load_kube_config``. Forwarding the path
    as a kwarg would be inert, so the variable must pass through the process
    environment untouched.
    """
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("KUBECONFIG", "/run/secrets/kix1.kubeconfig")
    monkeypatch.delenv("HARBOR_K8S_KUBECONFIG", raising=False)
    monkeypatch.delenv("HARBOR_K8S_NAMESPACE", raising=False)
    monkeypatch.delenv("HARBOR_K8S_IMAGE_MODE", raising=False)

    cfg = _config()

    assert cfg.environment_kwargs == {"namespace": "default", "image_mode": "prebuilt"}


def test_harbor_k8s_kubeconfig_is_rejected(monkeypatch):
    """A misnamed kubeconfig variable must fail loudly, not silently.

    ``BaseEnvironment.__init__`` ends in ``*args, **kwargs``, so an unknown
    ``kubeconfig=`` key is swallowed without a warning. Harbor would then fall
    back to ``~/.kube/config`` (or the in-cluster service account) and schedule
    every trial pod on whichever cluster that happens to name -- a wrong-cluster
    run that looks completely healthy in the logs.
    """
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")
    monkeypatch.setenv("HARBOR_K8S_KUBECONFIG", "/tmp/kix1.kubeconfig")

    with pytest.raises(ValueError, match="KUBECONFIG"):
        _config()


def test_harbor_k8s_kubeconfig_is_ignored_on_the_docker_path(monkeypatch):
    """The guard is scoped to the cluster path, so a stale var cannot break docker."""
    monkeypatch.delenv("HARBOR_ENV_TYPE", raising=False)
    monkeypatch.setenv("HARBOR_K8S_KUBECONFIG", "/tmp/kix1.kubeconfig")

    cfg = _config()

    assert cfg.environment_type == "docker"
    assert cfg.environment_kwargs == {}


def test_explicit_environment_type_overrides_env(monkeypatch):
    """A per-model override wins, so one model can opt out of the cluster."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")

    cfg = _config(environment_type="docker")

    assert cfg.environment_type == "docker"


def test_harbor_timeout_sec_parsed_as_float(monkeypatch):
    monkeypatch.setenv("HARBOR_TIMEOUT_SEC", "7200")

    assert _config().harbor_timeout_sec == 7200.0
