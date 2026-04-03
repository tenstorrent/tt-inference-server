// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/mock_device_pipeline.hpp"

#include <algorithm>
#include <chrono>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/debug.hpp"

namespace sp_pipeline {

namespace {
constexpr uint64_t FALLBACK_TOKEN_ID = 12345;
}

MockDevicePipeline::MockDevicePipeline(MockDeviceConfig config)
    : config(config) {
  pipelineThread = std::thread([this] { pipelineLoop(); });
}

MockDevicePipeline::~MockDevicePipeline() { exit(); }

void MockDevicePipeline::write(uint32_t taskId,
                               const std::vector<int64_t>& tokenIds,
                               uint32_t maxTokens, RequestPhase phase) {
  auto req = std::make_unique<PipelineRequest>();
  req->taskId = taskId;
  req->tokenIds = tokenIds;
  req->maxTokens = maxTokens;
  req->isDecode = (phase == RequestPhase::DECODE);

  std::unique_lock lock(inputMutex);
  inputNotFull.wait(lock, [this] {
    return inputQueue.size() < config.writeQueueCapacity ||
           stop.load(std::memory_order_relaxed);
  });
  if (stop.load(std::memory_order_relaxed)) return;
  inputQueue.push_back(std::move(req));
}

std::optional<llm_engine::TokenResult> MockDevicePipeline::read() {
  ZoneScopedN("MockDevice::read");
  std::unique_lock lock(outputMutex);
  outputNotEmpty.wait(lock, [this] {
    return !outputQueue.empty() || stop.load(std::memory_order_relaxed);
  });
  if (outputQueue.empty()) return std::nullopt;
  auto result = std::move(outputQueue.front());
  outputQueue.pop_front();
  return result;
}

void MockDevicePipeline::exit() {
  if (stop.exchange(true)) return;
  inputNotFull.notify_all();
  outputNotEmpty.notify_all();
  if (pipelineThread.joinable()) pipelineThread.join();
  LLM_ENGINE_LOG("mock_device_pipeline") << "exit" << std::endl;
}

// ---------------------------------------------------------------------------
// Pipeline internals — all called from pipeline_thread_ only,
// except the output queue push which is synchronised.
// ---------------------------------------------------------------------------

void MockDevicePipeline::drainInput() {
  std::lock_guard lock(inputMutex);
  while (!inputQueue.empty()) {
    auto& req = inputQueue.front();
    if (req->isDecode) {
      decodeQueue.push_back(std::move(req));
    } else {
      prefillQueue.push_back(std::move(req));
    }
    inputQueue.pop_front();
  }
  inputNotFull.notify_all();
}

void MockDevicePipeline::emitToken(RequestPtr& req) {
  uint64_t tokenId =
      req->tokensGenerated < req->tokenIds.size()
          ? static_cast<uint64_t>(req->tokenIds[req->tokensGenerated])
          : FALLBACK_TOKEN_ID;
  ++req->tokensGenerated;

  {
    std::lock_guard lock(outputMutex);
    outputQueue.emplace_back(req->taskId, tokenId);
  }
  outputNotEmpty.notify_one();
}

void MockDevicePipeline::handleCompletion(RequestPtr req) {
  ZoneScopedN("MockDevice::handle_completion");

  emitToken(req);

  if (!req->isDecode) {
    req->isDecode = true;
  }

  if (req->tokensGenerated < req->maxTokens) {
    decodeQueue.push_back(std::move(req));
  }
}

void MockDevicePipeline::insertInFlight(InFlightRequest entry) {
  auto it = std::lower_bound(inFlightPipeline.begin(), inFlightPipeline.end(),
                             entry.completeAtTick,
                             [](const InFlightRequest& e, size_t tick) {
                               return e.completeAtTick < tick;
                             });
  inFlightPipeline.insert(it, std::move(entry));
}

MockDevicePipeline::RequestPtr MockDevicePipeline::scheduleNext() {
  if (!decodeQueue.empty()) {
    auto req = std::move(decodeQueue.front());
    decodeQueue.pop_front();
    return req;
  }
  if (!prefillQueue.empty()) {
    auto req = std::move(prefillQueue.front());
    prefillQueue.pop_front();
    return req;
  }
  return nullptr;
}

void MockDevicePipeline::trySchedule() {
  auto next = scheduleNext();
  if (!next) return;

  activeReq = std::move(next);
  if (activeReq->isDecode) {
    feedRemaining = 1;
  } else {
    size_t tokensRemaining =
        activeReq->tokenIds.size() - activeReq->prefillOffset;
    size_t chunkTokens = std::min(config.prefillChunkSize, tokensRemaining);
    activeReq->prefillOffset += static_cast<uint32_t>(chunkTokens);
    feedRemaining = chunkTokens;
  }
}

void MockDevicePipeline::pipelineLoop() {
  using clock = std::chrono::steady_clock;
  const auto tickDuration = std::chrono::microseconds(config.stageDurationUs);

  while (!stop.load(std::memory_order_relaxed)) {
    auto tickStart = clock::now();

    drainInput();

    while (!inFlightPipeline.empty() &&
           inFlightPipeline.front().completeAtTick <= currentTick) {
      handleCompletion(std::move(inFlightPipeline.front().req));
      inFlightPipeline.pop_front();
    }

    if (activeReq && !activeReq->isDecode && !decodeQueue.empty()) {
      activeReq->prefillOffset -= feedRemaining;
      prefillQueue.push_front(std::move(activeReq));
      activeReq = nullptr;
      feedRemaining = 0;
    }

    if (!activeReq) {
      trySchedule();
    }

    if (activeReq) {
      --feedRemaining;
      if (feedRemaining == 0) {
        bool isIntermediatePrefill =
            !activeReq->isDecode &&
            activeReq->prefillOffset < activeReq->tokenIds.size();
        if (isIntermediatePrefill) {
          prefillQueue.push_back(std::move(activeReq));
        } else {
          insertInFlight(
              {currentTick + config.numStages, std::move(activeReq)});
        }
      }
    }

    ++currentTick;

    while ((clock::now() - tickStart) < tickDuration) {
    }
  }
}

}  // namespace sp_pipeline
