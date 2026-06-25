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
 * @brief Owns the *how* of peer discovery: resolve a worker's peers by name
 *        through the Transfer Engine's metadata service, blocking until every
 *        peer is found or the timeout elapses.
 *
 */
class PeerDiscoveryService {
 public:
  explicit PeerDiscoveryService(PeerDiscoveryConfig config = {});

  /**
   * @brief Resolve every peer name to a segment handle.
   * @param engine    initialised engine whose metadata service is queried.
   * @param peerNames logical segment names to resolve; empty => no peers.
   * @return name -> handle for all peers (empty map when @p peerNames is
   *         empty), or nullopt if the timeout elapsed before all were found.
   */
  std::optional<std::map<std::string, SegmentHandle>> discover(
      ITransferEngine& engine, const std::vector<std::string>& peerNames);

 private:
  /// The retry loop: poll openSegment() for unresolved names until the deadline.
  std::optional<std::map<std::string, SegmentHandle>> resolveAll(
      ITransferEngine& engine, const std::vector<std::string>& peerNames);

  PeerDiscoveryConfig config_;
};

}  // namespace tt::transport
