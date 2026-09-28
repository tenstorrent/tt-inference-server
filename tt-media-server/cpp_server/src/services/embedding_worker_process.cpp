// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "services/embedding_worker_process.hpp"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

#include "config/settings.hpp"
#include "services/embedding_pipe.hpp"
#include "utils/logger.hpp"

namespace tt::services::embedding_detail {

namespace {

bool createPipe(int wid, tt::utils::ScopedFd& readEnd,
                tt::utils::ScopedFd& writeEnd) {
  int raw[2] = {-1, -1};
  if (pipe(raw) < 0) {
    TT_LOG_ERROR("[EmbeddingService] Failed to create pipes for worker {}",
                 wid);
    return false;
  }
  readEnd = tt::utils::ScopedFd(raw[0]);
  writeEnd = tt::utils::ScopedFd(raw[1]);
  return true;
}

}  // namespace

bool WorkerProcess::spawn(
    int wid, std::function<void(int readFd, int writeFd)> childMain) {
  workerId = wid;

  tt::utils::ScopedFd reqRead, reqWrite;
  if (!createPipe(wid, reqRead, reqWrite)) return false;

  tt::utils::ScopedFd respRead, respWrite;
  if (!createPipe(wid, respRead, respWrite)) {
    return false;
  }

  pid_t child = fork();
  if (child < 0) {
    TT_LOG_ERROR("[EmbeddingService] Failed to fork worker {}", wid);
    return false;
  }

  if (child == 0) {
    reqWrite.reset();
    respRead.reset();
    childMain(reqRead.release(), respWrite.release());
    _exit(0);
  }

  reqRead.reset();
  respWrite.reset();
  pid.store(child);
  writeFd = std::move(reqWrite);
  readFd = std::move(respRead);
  running.store(true);

  TT_LOG_INFO(
      "[EmbeddingService] Spawned worker {} with PID {} "
      "(TT_VISIBLE_DEVICES={}) writeFd={} readFd={}",
      wid, child, tt::config::visibleDevicesForWorker(wid), writeFd.get(),
      readFd.get());
  return true;
}

bool WorkerProcess::checkAlive() {
  const pid_t p = pid.load();
  if (p <= 0) return false;
  int status;
  pid_t result = waitpid(p, &status, WNOHANG);
  if (result != p) return true;

  if (WIFEXITED(status)) {
    TT_LOG_ERROR("[EmbeddingService] Worker {} exited with code {}", workerId,
                 WEXITSTATUS(status));
  } else if (WIFSIGNALED(status)) {
    TT_LOG_ERROR("[EmbeddingService] Worker {} killed by signal {}", workerId,
                 WTERMSIG(status));
  }
  // The child is reaped; clear the pid so nobody signals a recycled one.
  pid.store(-1);
  isReady.store(false);
  return false;
}

bool WorkerProcess::sendRequest(const std::string& json) {
  if (!pipeWrite(writeFd.get(), json.data(), json.size())) {
    TT_LOG_ERROR("[EmbeddingService] Worker {} pipe write failed: {}", workerId,
                 strerror(errno));
    isReady.store(false);
    return false;
  }
  return true;
}

std::vector<uint8_t> WorkerProcess::receiveResponse() {
  auto buf = pipeReadBinary(readFd.get());
  if (buf.empty()) {
    TT_LOG_ERROR("[EmbeddingService] Worker {} response read failed", workerId);
    isReady.store(false);
  }
  return buf;
}

void WorkerProcess::terminate() {
  const pid_t p = pid.load();
  if (p > 0) {
    kill(p, SIGTERM);
    waitpid(p, nullptr, 0);
    pid.store(-1);
    TT_LOG_INFO("[EmbeddingService] Worker {} terminated", workerId);
  }
  writeFd.reset();
  readFd.reset();
}

}  // namespace tt::services::embedding_detail
