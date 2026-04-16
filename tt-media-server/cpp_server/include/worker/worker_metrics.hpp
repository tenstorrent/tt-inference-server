// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace tt::worker {

/**
 * Per-worker heartbeat tracking exposed via a dedicated /metrics endpoint.
 *
 * Heartbeat timestamps are stored as atomics and converted to age-in-seconds
 * at scrape time (renderText). This keeps the hot path (step/handleOutput)
 * to a single atomic store.
 *
 * The header is kept free of prometheus includes so it can be used from
 * llm_runner_lib without adding a prometheus dependency.
 */
class WorkerMetrics {
 public:
  static WorkerMetrics& instance();

  void initialize(int workerId);

  void updateStepHeartbeat();
  void updateOutputHeartbeat();
  void incrementActiveRequests();
  void decrementActiveRequests();

  std::string renderText();

  double stepAgeSec() const;
  double outputAgeSec() const;
  uint32_t activeRequests() const;

 private:
  WorkerMetrics() = default;

  static uint64_t nowMs();

  int workerId{0};
  bool initialized{false};
  std::atomic<uint64_t> stepEpochMs{0};
  std::atomic<uint64_t> lastOutputEpochMs{0};
  std::atomic<uint32_t> activeRequestsCount{0};
};

}  // namespace tt::worker
