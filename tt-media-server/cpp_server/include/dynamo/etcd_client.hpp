// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Minimal etcd v3 client over the JSON HTTP gateway.
 *
 * etcd's gRPC-gateway exposes the v3 KV/Lease API as JSON HTTP endpoints on
 * the same port as gRPC (default 2379). This client speaks HTTP/1.1 directly
 * over a POSIX socket so we don't pull in libcurl or grpc.
 *
 * Only the surface used by Dynamo discovery is implemented:
 *   - LeaseGrant: create a lease the backend will associate with its keys.
 *   - LeaseKeepAlive: refresh the lease so etcd doesn't reap our entries.
 *   - LeaseRevoke: explicit deregistration on shutdown.
 *   - Put / DeleteRange: write / remove the instance + MDC documents.
 *
 * HTTPS endpoints are NOT supported (frontends in the same trust domain
 * typically use plain HTTP; production deployments should terminate TLS at a
 * sidecar proxy).
 */

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tt::dynamo {

/// Throws when the etcd HTTP gateway responds with anything other than 200,
/// or when the JSON body cannot be parsed. Callers translate this into a
/// retry / re-register on the next keep-alive tick.
class EtcdError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class EtcdClient {
 public:
  /**
   * `endpoint` may be a single `http://host:port` URL or a comma-separated
   * list (only the first is used; multi-endpoint failover is not
   * implemented). `timeout_ms` bounds connect + read for every call.
   */
  explicit EtcdClient(const std::string& endpoint, int timeout_ms = 5000);

  /// Grant a new lease with the given TTL (seconds). Returns the lease id.
  int64_t leaseGrant(int64_t ttl_secs);

  /// Refresh `lease_id`. Returns the TTL etcd reports remaining (in
  /// seconds); zero means etcd considers the lease already expired.
  int64_t leaseKeepAlive(int64_t lease_id);

  /// Revoke `lease_id`. All keys attached to it are deleted atomically.
  void leaseRevoke(int64_t lease_id);

  /// Put a single key-value pair, optionally attached to a lease (0 = none).
  void put(const std::string& key, const std::string& value,
           int64_t lease_id = 0);

  /// Delete a single key (range_end is left unset).
  void deleteRange(const std::string& key);

 private:
  std::string host_;
  int port_ = 2379;
  int timeout_ms_;
};

}  // namespace tt::dynamo
