// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace tt::gateway {

struct GatewayPrefillMetricSnapshot {
  std::string server_id;
  bool healthy = false;
  bool accepting_tasks = false;
  uint32_t in_flight = 0;
  size_t cached_blocks = 0;
  double heartbeat_age_seconds = 0.0;
};

class GatewayMetrics {
 public:
  static GatewayMetrics& instance();

  GatewayMetrics(const GatewayMetrics&) = delete;
  GatewayMetrics& operator=(const GatewayMetrics&) = delete;
  ~GatewayMetrics();

  void recordRoutingDecision(std::string_view reason);
  void observePrefixMatchDepth(size_t depth);
  void setRoutingTableSize(size_t size);

  void recordRequestCompleted(std::string_view serverId,
                              std::string_view outcome,
                              std::chrono::steady_clock::duration latency);
  void recordRequestFailed(std::string_view reason);
  void recordCancel(bool sent);
  void recordTimeout(std::string_view serverId);
  void recordPrefillDownTasks(size_t count);
  void recordCacheBlocksAdded(size_t count);
  void recordCacheBlocksEvicted(size_t count);

  void setDecodeConnected(bool connected);
  void setPrefillSnapshots(
      std::span<const GatewayPrefillMetricSnapshot> snapshots);

  std::string renderText() const;
  void resetForTests();

 private:
  GatewayMetrics();

  class Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace tt::gateway
