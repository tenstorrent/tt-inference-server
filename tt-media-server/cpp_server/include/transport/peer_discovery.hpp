// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "transport/transfer_types.hpp"

namespace tt::transport {

class ITransferEngine;

/// Tunables for the discovery retry loop.
struct PeerDiscoveryConfig {
  int poll_interval_ms = 1000;  ///< Delay between resolve sweeps.
  int timeout_sec = 30;         ///< Give up if not all peers resolve in time.
};

/**
 * @brief Resolves a worker's peers by name through the Transfer Engine's
 *        metadata service, blocking until every peer is found.
 *
 * Encapsulates the discovery behaviour proven in the #4209 PoC
 * (tests/integration/migration_worker_discovery.cpp): poll openSegment() for
 * each peer, retrying only the unresolved ones, until all are reachable or the
 * timeout elapses. Owned and driven by MooncakeMigrationWorker on bring-up.
 */
class PeerDiscovery {
 public:
  explicit PeerDiscovery(PeerDiscoveryConfig config = {});

  /**
   * @brief Resolve every peer name to a segment handle.
   * @return name -> handle for all peers, or nullopt if the timeout elapsed
   *         before all were found.
   */
  std::optional<std::map<std::string, SegmentHandle>> resolveAll(
      ITransferEngine& engine, const std::vector<std::string>& peerNames);

 private:
  PeerDiscoveryConfig config_;
};

}  // namespace tt::transport
