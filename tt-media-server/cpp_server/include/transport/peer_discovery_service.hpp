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

/// Tunables for the discovery retry loop. Both fields are sanitised by the
/// PeerDiscoveryService constructor (see clamping rules there).
struct PeerDiscoveryConfig {
  int poll_interval_ms = 1000;  ///< Delay between resolve sweeps (>= 1).
  int timeout_sec = 30;         ///< Give up if not all peers resolve in time.
};

/**
 * @brief Owns the *how* of peer discovery: resolve a worker's peers by name
 *        through the Transfer Engine's metadata service, blocking until every
 *        peer is found or the timeout elapses.
 *
 * Stateless and re-entrant once constructed: discover() only reads the
 * (sanitised) config and operates on the engine handed to it, so a single
 * instance may be reused or shared across calls. The peer list is treated
 * defensively — duplicates and empty names are ignored — so a sloppy caller
 * cannot wedge discovery into a permanent false timeout.
 */
class PeerDiscoveryService {
 public:
  /// Sanitises @p config: poll_interval_ms is clamped to >= 1 (a non-positive
  /// interval would busy-spin the metadata service) and a negative timeout is
  /// treated as 0. Both clamps log a warning.
  explicit PeerDiscoveryService(PeerDiscoveryConfig config = {});

  /**
   * @brief Resolve every peer name to a segment handle.
   * @param engine    initialised engine whose metadata service is queried.
   * @param peerNames logical segment names to resolve; duplicates and empty
   *                  names are ignored, and an empty (or all-empty) list is a
   *                  no-op success.
   * @return name -> handle for all unique peers (empty map when there are no
   *         valid names), or nullopt if the timeout elapsed before all were
   *         found.
   */
  std::optional<std::map<std::string, SegmentHandle>> discover(
      ITransferEngine& engine, const std::vector<std::string>& peerNames);

 private:
  /// Poll sweeps until all @p wanted are resolved or the deadline passes;
  /// returns whatever was resolved (possibly partial). Pure: no outcome logging.
  std::map<std::string, SegmentHandle> resolveAll(
      ITransferEngine& engine, const std::vector<std::string>& wanted) const;

  /// One pass over the not-yet-resolved names, adding any newly found handles.
  void sweepUnresolved(ITransferEngine& engine,
                       const std::vector<std::string>& wanted,
                       std::map<std::string, SegmentHandle>& resolved) const;

  PeerDiscoveryConfig config_;
};

}  // namespace tt::transport
