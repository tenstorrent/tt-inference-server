// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Discovery registration for Dynamo backends.
 *
 * Uses etcd: PUTs instance + Model Descriptor Card JSON under keys
 * `v1/instances/<key>` and `v1/mdc/<key>` attached to a lease. Keep-alive
 * refreshes that lease; unregister revokes it (atomically deleting both keys).
 *
 * Wire-compatible with NVIDIA Dynamo's KVStoreDiscovery; the frontend reads
 * the same JSON payloads.
 */

#include <cstdint>
#include <memory>
#include <string>

namespace tt::dynamo {

struct DiscoveryConfig {
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

  /// Refresh the registration so it doesn't expire. This is a lease
  /// keep-alive that falls back to a full re-register if the lease has been
  /// reaped. Errors are swallowed (logged) so a transient outage doesn't
  /// kill the worker.
  virtual void keepAlive() = 0;

  /// Remove the registration. Errors are swallowed (best effort on
  /// shutdown).
  virtual void unregisterSelf() = 0;

  /// Suggested keep-alive period. The endpoint thread sleeps for this many
  /// seconds between `keepAlive()` calls.
  virtual int keepAliveIntervalSecs() const = 0;
};

}  // namespace tt::dynamo
