// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Discovery registration for Dynamo backends.
 *
 * Two backends, both wire-compatible with NVIDIA Dynamo (the frontend reads the
 * same JSON payloads regardless):
 *   - Etcd: PUTs instance + Model Descriptor Card JSON under keys
 *     `v1/instances/<key>` and `v1/mdc/<key>` attached to a lease. Keep-alive
 *     refreshes the lease; unregister revokes it.
 *   - Kubernetes: server-side-applies a `DynamoWorkerMetadata` custom resource
 *     that bundles the same per-instance JSON into `spec.data`. No lease —
 *     liveness/removal is owned by pod readiness (EndpointSlices) and the CR's
 *     Pod owner-reference (garbage collection).
 */

#include <cstdint>
#include <memory>
#include <string>

namespace tt::dynamo {

/// Which discovery backend a registration talks to. Matches the frontend's
/// DYN_DISCOVERY_BACKEND.
enum class DiscoveryBackend { ETCD, KUBERNETES };

struct DiscoveryConfig {
  /// Selects the backend `DiscoveryRegistration::create` instantiates.
  DiscoveryBackend backend = DiscoveryBackend::ETCD;

  // ---- Etcd backend ------------------------------------------------------
  /// Single etcd HTTP endpoint or comma-separated list (only the first is
  /// dialed today). Use `http://<host>:2379`; HTTPS is unsupported.
  std::string etcd_endpoints = "http://localhost:2379";
  /// Lease TTL the backend grants for its instance + MDC keys. Keep-alive
  /// is performed at half this interval so brief frontend hiccups don't
  /// cause an etcd reap.
  int64_t etcd_lease_ttl_secs = 10;

  // ---- Identity ----------------------------------------------------------
  std::string namespace_name = "default";
  std::string component = "backend";
  std::string endpoint = "generate";
  std::string instance_id_hex;
  uint64_t instance_id = 0;
  /// "host:port/instance_id_hex/endpoint" — the address Dynamo dials to send
  /// us a request.
  std::string tcp_address;
  /// Display name in /v1/models (e.g. "deepseek-ai/DeepSeek-R1-0528").
  std::string model_name;
  /// Filesystem directory containing config.json + tokenizer.json +
  /// tokenizer_config.json.
  std::string model_path;

  // ---- Kubernetes backend ------------------------------------------------
  /// API server base URL, e.g. "https://10.0.0.1:443".
  std::string kube_api_server;
  /// Path to the ServiceAccount bearer token file.
  std::string kube_token_path =
      "/var/run/secrets/kubernetes.io/serviceaccount/token";
  /// Validate the API server's TLS certificate.
  bool kube_validate_cert = true;
  /// Namespace the DynamoWorkerMetadata CR is created in.
  std::string pod_namespace = "default";
  /// Pod identity for the CR name and owner reference (downward API).
  std::string pod_name;
  std::string pod_uid;
  /// Name of the DynamoWorkerMetadata CR. Pod mode: equal to `pod_name`.
  std::string cr_name;
};

/**
 * Lifecycle handle for a discovery registration. Construct via
 * `DiscoveryRegistration::create`, call `registerSelf()` once after binding,
 * `keepAlive()` periodically, and `unregisterSelf()` on shutdown.
 *
 * Implementations are NOT thread-safe; callers must serialize access (the
 * existing keep-alive thread does).
 */
class DiscoveryRegistration {
 public:
  static std::unique_ptr<DiscoveryRegistration> create(
      const DiscoveryConfig& config);

  virtual ~DiscoveryRegistration() = default;

  /// Publish the instance + MDC documents. Throws on transport failure.
  virtual void registerSelf() = 0;

  /// Refresh the registration so it doesn't expire (e.g. etcd lease renewal,
  /// falling back to a full re-register if the lease was reaped). Only invoked
  /// when `keepAliveIntervalSecs() > 0`; the default no-op suits backends with
  /// no expiry (kubernetes, whose liveness Kubernetes owns). Implementations
  /// should swallow/log errors so a transient outage doesn't kill the worker.
  virtual void keepAlive() {}

  /// Remove the registration. Errors are swallowed (best effort on
  /// shutdown).
  virtual void unregisterSelf() = 0;

  /// Keep-alive period: the endpoint thread sleeps this many seconds between
  /// `keepAlive()` calls. The default 0 means the backend needs no periodic
  /// keep-alive (kubernetes, whose liveness Kubernetes owns) and the endpoint
  /// spawns no keep-alive thread; override to a positive value for lease-based
  /// backends (etcd).
  virtual int keepAliveIntervalSecs() const { return 0; }
};

}  // namespace tt::dynamo
