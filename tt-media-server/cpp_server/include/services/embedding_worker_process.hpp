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
 * One forked embedding worker OS process, as seen from the parent.
 *
 * Pipe ownership: the parent holds writeFd (request pipe, write end) and
 * readFd (response pipe, read end); spawn() hands the opposite ends to the
 * child and closes them in the parent. Reaping ownership: waitpid is called
 * only by the worker's dispatch thread (checkAlive) and by terminate();
 * anyone else probing liveness must use kill(pid, 0) so the exit status is
 * not consumed behind the dispatch thread's back.
 */
struct WorkerProcess {
  int workerId = -1;
  /// Atomic because health snapshots read it while the startup thread spawns.
  std::atomic<pid_t> pid{-1};
  tt::utils::ScopedFd writeFd;  // parent → child (request pipe write end)
  tt::utils::ScopedFd readFd;   // child → parent (response pipe read end)
  std::atomic<bool> isReady{false};
  std::atomic<bool> running{false};
  std::unique_ptr<std::thread> dispatchThread;

  /** Fork the worker. The child runs childMain(readFd, writeFd) and never
   * returns; the parent takes ownership of its pipe ends and marks the
   * worker running (NOT ready: isReady flips only on the READY sentinel). */
  bool spawn(int wid, std::function<void(int readFd, int writeFd)> childMain);

  /** Non-blocking liveness probe; reaps and logs the exit status if the
   * child has died. Called from the worker's dispatch thread only. */
  bool checkAlive();

  /** Length-prefixed write of one request batch; clears isReady on failure.
   */
  bool sendRequest(const std::string& json);

  /** Length-prefixed read of one response batch; clears isReady on failure.
   */
  std::vector<uint8_t> receiveResponse();

  /** SIGTERM + blocking waitpid, then closes both pipe ends. */
  void terminate();
};

}  // namespace tt::services::embedding_detail
