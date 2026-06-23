// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/peer_discovery.hpp"

#include <chrono>
#include <thread>

#include "transport/i_transfer_engine.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

PeerDiscovery::PeerDiscovery(PeerDiscoveryConfig config) : config_(config) {}

std::optional<std::map<std::string, SegmentHandle>> PeerDiscovery::resolveAll(
    ITransferEngine& engine, const std::vector<std::string>& peerNames) {
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
        TT_LOG_DEBUG("[PeerDiscovery] resolved '{}' ({}/{})", name,
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
        "[PeerDiscovery] timed out after {}s: resolved {}/{} peers; still "
        "missing: {}",
        config_.timeout_sec, resolved.size(), peerNames.size(), missing);
    return std::nullopt;
  }
  return resolved;
}

}  // namespace tt::transport
