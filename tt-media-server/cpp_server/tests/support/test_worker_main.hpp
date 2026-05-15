// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Worker-subprocess entry point for the integration test binary.
// WorkerManager re-execs the test binary with "--worker <id>"; main() routes
// that to runWorkerSubprocess. The subprocess signals warmup and idles until
// SIGTERM — the test process owns the memory-manager and task/result queues.

#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

#include "config/settings.hpp"
#include "ipc/boost/boost_warmup_signal_queue.hpp"
#include "utils/logger.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"

namespace tt::test {

inline int runWorkerSubprocess(int workerId) {
  tt::utils::ZeroOverheadLogger::initialize();

  tt::worker::SingleProcessWorkerMetrics::instance().initialize(
      workerId, tt::worker::MetricsLayout::SP_PIPELINE_RUNNER);

  // Signal warmup: parent unblocks isModelReady() only after this.
  tt::ipc::boost::WarmupSignalQueue warmupQueue(
      tt::config::ttWarmupSignalsQueueName());
  warmupQueue.sendReady(workerId);

  static std::atomic<bool> done{false};
  std::signal(SIGTERM, [](int) { done.store(true); });
  std::signal(SIGINT, [](int) { done.store(true); });
  while (!done.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return 0;
}

}  // namespace tt::test
