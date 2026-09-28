// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <sys/types.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils/scoped_fd.hpp"

namespace tt::services::embedding_detail {

/**
 * One forked embedding worker, as seen from the parent. The parent owns
 * writeFd/readFd; the child gets the opposite pipe ends. Only checkAlive()
 * and terminate() may waitpid(); everyone else probes with kill(pid, 0).
 */
struct WorkerProcess {
  int workerId = -1;
  /// Atomic: health snapshots read it while the startup thread spawns.
  std::atomic<pid_t> pid{-1};
  tt::utils::ScopedFd writeFd;  // parent → child (request pipe write end)
  tt::utils::ScopedFd readFd;   // child → parent (response pipe read end)
  std::atomic<bool> isReady{false};
  std::atomic<bool> running{false};
  std::unique_ptr<std::thread> dispatchThread;

  /** Fork the worker; the child runs childMain(readFd, writeFd) and never
   * returns. isReady stays false until the READY sentinel arrives. */
  bool spawn(int wid, std::function<void(int readFd, int writeFd)> childMain);

  /** Non-blocking liveness probe; reaps and clears pid if the child died. */
  bool checkAlive();

  /** Writes one framed request batch; clears isReady on failure. */
  bool sendRequest(const std::string& json);

  /** Reads one framed response batch; clears isReady on failure. */
  std::vector<uint8_t> receiveResponse();

  /** SIGTERM + blocking waitpid, then closes both pipe ends. */
  void terminate();
};

}  // namespace tt::services::embedding_detail
