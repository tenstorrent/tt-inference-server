// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <sys/wait.h>

#include <atomic>
#include <cstring>

#include "utils/logger.hpp"
#include "worker/worker_manager.hpp"

namespace tt::worker {

ProcessLivenessTransition pollProcessLiveness(pid_t pid,
                                              std::atomic<bool>& aliveFlag,
                                              size_t workerIdx) {
  if (pid <= 0) {
    bool wasAlive = aliveFlag.exchange(false);
    return {/*stillAlive=*/false, /*transitionedToDead=*/wasAlive};
  }
  int status;
  pid_t result = waitpid(pid, &status, WNOHANG);
  if (result == 0) {
    return {/*stillAlive=*/true, /*transitionedToDead=*/false};
  }
  if (result == pid) {
    bool wasAlive = aliveFlag.exchange(false);
    if (WIFSIGNALED(status)) {
      int sig = WTERMSIG(status);
      TT_LOG_CRITICAL(
          "[WorkerManager] Worker {} (PID {}) killed by signal {} ({})"
          "{}",
          workerIdx, pid, sig, strsignal(sig),
          WCOREDUMP(status) ? " -- core dumped" : "");
    } else if (WIFEXITED(status)) {
      int exitCode = WEXITSTATUS(status);
      if (exitCode != 0) {
        TT_LOG_CRITICAL(
            "[WorkerManager] Worker {} (PID {}) exited with code {}", workerIdx,
            pid, exitCode);
      } else {
        TT_LOG_WARN(
            "[WorkerManager] Worker {} (PID {}) exited normally (code 0)",
            workerIdx, pid);
      }
    } else {
      TT_LOG_CRITICAL(
          "[WorkerManager] Worker {} (PID {}) terminated, raw status=0x{:x}",
          workerIdx, pid, status);
    }
    return {/*stillAlive=*/false, /*transitionedToDead=*/wasAlive};
  }
  // waitpid returned an error (e.g. ECHILD because the process was reaped
  // elsewhere); err on the side of "alive" so a single failed waitpid does
  // not spuriously fire the death callback.
  return {/*stillAlive=*/true, /*transitionedToDead=*/false};
}

}  // namespace tt::worker
