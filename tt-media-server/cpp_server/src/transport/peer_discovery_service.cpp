// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/peer_discovery_service.hpp"

#include <chrono>
#include <thread>
#include <unordered_set>

#include "transport/i_transfer_engine.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {
constexpr int K_MIN_POLL_INTERVAL_MS = 1;

// Drop empty names and de-duplicate, preserving first-seen order so logs and
// resolution are deterministic. A duplicate name would otherwise inflate the
// expected count and wedge discovery into a permanent false timeout.
std::vector<std::string> uniquePeers(const std::vector<std::string>& names) {
  std::vector<std::string> unique;
  unique.reserve(names.size());
  std::unordered_set<std::string> seen;
  for (const auto& name : names) {
    if (name.empty()) continue;
    if (seen.insert(name).second) unique.push_back(name);
  }
  return unique;
}

// Comma-joined names in `wanted` that are absent from `resolved` — naming the
// stragglers makes a 20-worker mesh debuggable from a single log line.
std::string joinMissing(const std::vector<std::string>& wanted,
                        const std::map<std::string, SegmentHandle>& resolved) {
  std::string missing;
  for (const auto& name : wanted) {
    if (resolved.count(name)) continue;
    if (!missing.empty()) missing += ", ";
    missing += name;
  }
  return missing;
}
}  // namespace

PeerDiscoveryService::PeerDiscoveryService(PeerDiscoveryConfig config)
    : config_(config) {
  if (config_.poll_interval_ms < K_MIN_POLL_INTERVAL_MS) {
    TT_LOG_WARN(
        "[PeerDiscoveryService] poll_interval_ms {} too low; clamping to {}",
        config_.poll_interval_ms, K_MIN_POLL_INTERVAL_MS);
    config_.poll_interval_ms = K_MIN_POLL_INTERVAL_MS;
  }
  if (config_.timeout_sec < 0) {
    TT_LOG_WARN("[PeerDiscoveryService] negative timeout_sec {}; treating as 0",
                config_.timeout_sec);
    config_.timeout_sec = 0;
  }
}

std::optional<std::map<std::string, SegmentHandle>>
PeerDiscoveryService::discover(ITransferEngine& engine,
                               const std::vector<std::string>& peerNames,
                               const std::atomic<bool>* cancelToken) {
  const std::vector<std::string> wanted = uniquePeers(peerNames);
  if (wanted.size() != peerNames.size()) {
    TT_LOG_WARN(
        "[PeerDiscoveryService] ignored {} duplicate/empty peer name(s)",
        peerNames.size() - wanted.size());
  }
  if (wanted.empty()) {
    TT_LOG_WARN("[PeerDiscoveryService] no peers configured");
    return std::map<std::string, SegmentHandle>{};
  }

  auto resolved = resolveAll(engine, wanted, cancelToken);
  if (resolved.size() < wanted.size()) {
    const std::string missing = joinMissing(wanted, resolved);
    if (cancelToken && cancelToken->load()) {
      TT_LOG_WARN(
          "[PeerDiscoveryService] cancelled: resolved {}/{} peers; abandoned: "
          "{}",
          resolved.size(), wanted.size(), missing);
    } else {
      TT_LOG_ERROR(
          "[PeerDiscoveryService] timed out after {}s: resolved {}/{} peers; "
          "still missing: {}",
          config_.timeout_sec, resolved.size(), wanted.size(), missing);
    }
    return std::nullopt;
  }

  TT_LOG_INFO("[PeerDiscoveryService] CONNECTED to {} peers", resolved.size());
  return resolved;
}

std::map<std::string, SegmentHandle> PeerDiscoveryService::resolveAll(
    ITransferEngine& engine, const std::vector<std::string>& wanted,
    const std::atomic<bool>* cancelToken) const {
  std::map<std::string, SegmentHandle> resolved;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(config_.timeout_sec);

  // do/while guarantees at least one attempt — a 0s timeout means "try once",
  // never "skip discovery entirely".
  do {
    sweepUnresolved(engine, wanted, resolved);
    if (resolved.size() >= wanted.size()) break;
    if (cancelToken && cancelToken->load()) break;
    if (std::chrono::steady_clock::now() >= deadline) break;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config_.poll_interval_ms));
  } while (true);
  return resolved;
}

void PeerDiscoveryService::sweepUnresolved(
    ITransferEngine& engine, const std::vector<std::string>& wanted,
    std::map<std::string, SegmentHandle>& resolved) const {
  for (const auto& name : wanted) {
    if (resolved.count(name)) continue;
    const SegmentHandle handle = engine.openSegment(name);
    if (handle == kInvalidSegment) continue;
    resolved.emplace(name, handle);
    TT_LOG_DEBUG("[PeerDiscoveryService] resolved '{}' ({}/{})", name,
                 resolved.size(), wanted.size());
  }
}

}  // namespace tt::transport
