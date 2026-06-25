// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/peer_discovery_service.hpp"

#include <chrono>
#include <thread>

#include "transport/i_transfer_engine.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

PeerDiscoveryService::PeerDiscoveryService(PeerDiscoveryConfig config)
    : config_(config) {}

std::optional<std::map<std::string, SegmentHandle>>
PeerDiscoveryService::discover(ITransferEngine& engine,
                               const std::vector<std::string>& peerNames) {
  if (peerNames.empty()) {
    TT_LOG_WARN("[PeerDiscoveryService] no peers configured");
    return std::map<std::string, SegmentHandle>{};
  }

  auto resolved = resolveAll(engine, peerNames);
  if (!resolved) {
    TT_LOG_ERROR(
        "[PeerDiscoveryService] discovery timed out before all peers were "
        "reachable");
    return std::nullopt;
  }

  TT_LOG_INFO("[PeerDiscoveryService] CONNECTED to {} peers", resolved->size());
  return resolved;
}

std::optional<std::map<std::string, SegmentHandle>>
PeerDiscoveryService::resolveAll(ITransferEngine& engine,
                                 const std::vector<std::string>& peerNames) {
  std::map<std::string, SegmentHandle> resolved;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(config_.timeout_sec);

  while (resolved.size() < peerNames.size() &&
         std::chrono::steady_clock::now() < deadline) {
    for (const auto& name : peerNames) {
      if (resolved.count(name)) continue;
      const SegmentHandle handle = engine.openSegment(name);
      if (handle != kInvalidSegment) {
        resolved.emplace(name, handle);
        TT_LOG_DEBUG("[PeerDiscoveryService] resolved '{}' ({}/{})", name,
                     resolved.size(), peerNames.size());
      }
    }
    if (resolved.size() < peerNames.size()) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(config_.poll_interval_ms));
    }
  }

  if (resolved.size() < peerNames.size()) {
    // Name the peers we never reached — debugging a 20-worker mesh from a bare
    // "resolved 18/20" is painful otherwise.
    std::string missing;
    for (const auto& name : peerNames) {
      if (resolved.count(name)) continue;
      if (!missing.empty()) missing += ", ";
      missing += name;
    }
    TT_LOG_WARN(
        "[PeerDiscoveryService] timed out after {}s: resolved {}/{} peers; "
        "still missing: {}",
        config_.timeout_sec, resolved.size(), peerNames.size(), missing);
    return std::nullopt;
  }
  return resolved;
}

}  // namespace tt::transport
