// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Worker-subprocess entry point for the integration test binary.
// WorkerManager re-execs the test binary with "--worker <id>"; main() routes
// that to runWorkerSubprocess, which signals warmup, runs the memory manager
// loop for KV-cache slot allocation, and waits for SIGTERM.

#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <thread>

#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "services/memory_services/memory_manager.hpp"
#include "utils/logger.hpp"
#include "worker/single_process_worker_metrics.hpp"
#include "worker/worker_metrics_shm.hpp"

namespace tt::test {

inline int runWorkerSubprocess(int workerId) {
  tt::utils::ZeroOverheadLogger::initialize();

  tt::worker::SingleProcessWorkerMetrics::instance().initialize(
      workerId, tt::worker::MetricsLayout::SP_PIPELINE_RUNNER);

  // Initialize memory manager before signaling warmup — no requests will
  // arrive before the parent marks the server ready. The base MemoryManager
  // returns SUCCESS for ALLOCATE with an opaque slotId; that's all the test
  // needs — it isn't exercising real KV-cache pool sizing.
  tt::services::MemoryManager memMgr;

  // Signal warmup: parent unblocks isModelReady() only after this.
  tt::ipc::BoostIpcWarmupSignalQueue warmupQueue(
      tt::ipc::WARMUP_SIGNALS_QUEUE_NAME);
  warmupQueue.sendReady(workerId);

  static std::atomic<bool> done{false};
  std::signal(SIGTERM, [](int) { done.store(true); });
  std::signal(SIGINT, [](int) { done.store(true); });
  while (!done.load()) {
    auto req = memMgr.getRequest();
    if (req.has_value()) {
      memMgr.handleRequest(*req);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  return 0;
}

}  // namespace tt::test
