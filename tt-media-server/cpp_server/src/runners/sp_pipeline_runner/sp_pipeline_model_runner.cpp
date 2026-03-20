// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_model_runner.hpp"

#include <csignal>
#include <fstream>
#include <thread>

#include "runners/llm_runner/debug.hpp"
#include "utils/logger.hpp"

namespace sp_pipeline {

static constexpr const char* PIPELINE_READY_SENTINEL = "/dev/shm/tt_pipeline_ready";
static constexpr int INITIAL_CONNECT_WAIT_MS = 1000;
static constexpr int MAX_CONNECT_WAIT_MS = 30000;
static constexpr auto SENTINEL_CHECK_INTERVAL = std::chrono::seconds(5);

SpPipelineModelRunner::SpPipelineModelRunner(DecodeCallback callback)
    : decodeCallback(std::move(callback)),
      shmNames(),
      deviceInput(shmNames.write),
      deviceOutput(shmNames.read) {}

SpPipelineModelRunner::~SpPipelineModelRunner() { exit(); }

void SpPipelineModelRunner::connect() {
  int waitMs = INITIAL_CONNECT_WAIT_MS;
  int attempt = 0;

  while (!stop.load(std::memory_order_relaxed)) {
    try {
      deviceInput.close();
      deviceOutput.close();
      deviceInput.open();
      deviceOutput.open();
      connected_.store(true, std::memory_order_release);
      if (readerThread.joinable()) {
        readerThread.join();
      }
      readerThread = std::thread([this] { readerLoop(); });
      LLM_ENGINE_LOG("sp_pipeline") << "connected to model pipeline shm" << std::endl;
      return;
    } catch (const std::runtime_error& e) {
      ++attempt;
      TT_LOG_INFO(
          "SpPipelineModelRunner: waiting for pipeline shm (attempt {}, "
          "retry in {}ms): {}",
          attempt, waitMs, e.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(waitMs));
      waitMs = std::min(waitMs * 2, MAX_CONNECT_WAIT_MS);
    }
  }
}

bool SpPipelineModelRunner::isConnected() const {
  return connected_.load(std::memory_order_acquire);
}

void SpPipelineModelRunner::write(const std::string& taskId,
                                  const std::vector<int64_t>& tokenIds,
                                  uint32_t maxTokens, RequestPhase /*phase*/) {
  if (!connected_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "SpPipelineModelRunner: write called while not connected to pipeline");
  }
  // TODO: propagate phase to the shared-memory protocol for disaggregated mode.
  deviceInput.write(taskId, tokenIds, maxTokens);
}

void SpPipelineModelRunner::exit() {
  if (stop.exchange(true)) return;
  connected_.store(false, std::memory_order_release);
  if (readerThread.joinable()) readerThread.join();
  LLM_ENGINE_LOG("sp_pipeline") << "model runner exit" << std::endl;
}

bool SpPipelineModelRunner::isPipelineAlive() {
  std::ifstream f(PIPELINE_READY_SENTINEL);
  if (!f.is_open()) return false;
  pid_t pid = 0;
  f >> pid;
  if (!f || pid <= 0) return false;
  return kill(pid, 0) == 0 || errno == EPERM;
}

void SpPipelineModelRunner::readerLoop() {
  ReadResult readBuf;
  auto lastSentinelCheck = std::chrono::steady_clock::now();

  while (!stop.load(std::memory_order_relaxed)) {
    if (deviceOutput.tryRead(readBuf)) {
      llm_engine::TaskID tid = llm_engine::TaskID::ipcDeserialize(
          readBuf.taskId.data(), llm_engine::TaskID::K_SERIALIZED_SIZE);
      uint64_t tokenId = readBuf.tokenIds.empty() ? 0 : readBuf.tokenIds[0];
      llm_engine::TokenResult result(std::move(tid), tokenId);
      decodeCallback(result);
    } else {
      std::this_thread::yield();
      auto now = std::chrono::steady_clock::now();
      if (now - lastSentinelCheck >= SENTINEL_CHECK_INTERVAL) {
        lastSentinelCheck = now;
        if (!isPipelineAlive()) {
          TT_LOG_WARN(
              "SpPipelineModelRunner: pipeline process gone "
              "(sentinel PID check failed), disconnecting");
          connected_.store(false, std::memory_order_release);
          return;
        }
      }
    }
  }
}

}  // namespace sp_pipeline
