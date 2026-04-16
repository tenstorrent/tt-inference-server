// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace tt::worker {

/**
 * Minimal HTTP server for the worker /metrics endpoint.
 *
 * Uses plain POSIX sockets instead of Drogon to avoid interference with
 * the main process's auto-registered controllers (Drogon's app() is a
 * global singleton that initializes all controllers on run()).
 *
 * Accepts connections, responds with WorkerMetrics::renderText() for any
 * GET request, and closes the connection. Non-blocking accept loop with
 * a stop flag for clean shutdown.
 */
class WorkerMetricsServer {
 public:
  WorkerMetricsServer(int workerId, uint16_t port);
  ~WorkerMetricsServer();

  WorkerMetricsServer(const WorkerMetricsServer&) = delete;
  WorkerMetricsServer& operator=(const WorkerMetricsServer&) = delete;

  bool start();
  void stop();

 private:
  void loop();

  int workerId;
  uint16_t port;
  int listenFd{-1};
  std::atomic<bool> running{false};
  std::thread thread;
};

}  // namespace tt::worker
