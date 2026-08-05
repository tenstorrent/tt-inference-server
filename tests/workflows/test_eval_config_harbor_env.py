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
    monkeypatch.setenv("HARBOR_K8S_KUBECONFIG", "/tmp/kubeconfig")
    monkeypatch.setenv("HARBOR_K8S_CONTEXT", "kix1")
    monkeypatch.setenv("HARBOR_K8S_IMAGE_REGISTRY", "registry.local/harbor")
    monkeypatch.setenv("HARBOR_K8S_IMAGE_PULL_SECRET", "regcred")
    monkeypatch.setenv("HARBOR_K8S_SERVICE_ACCOUNT", "harbor-task")

    cfg = _config()

    assert cfg.environment_kwargs == {
        "namespace": "harbor-kube-env",
        "image_mode": "build-and-push",
        "kubeconfig": "/tmp/kubeconfig",
        "context": "kix1",
        "image_registry": "registry.local/harbor",
        "image_pull_secret": "regcred",
        "service_account": "harbor-task",
    }


def test_explicit_environment_type_overrides_env(monkeypatch):
    """A per-model override wins, so one model can opt out of the cluster."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "kubernetes")

    cfg = _config(environment_type="docker")

    assert cfg.environment_type == "docker"


def test_harbor_timeout_sec_parsed_as_float(monkeypatch):
    monkeypatch.setenv("HARBOR_TIMEOUT_SEC", "7200")

    assert _config().harbor_timeout_sec == 7200.0
