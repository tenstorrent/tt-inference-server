// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/tool_calls/tool_choice.hpp"
#include "ipc/interface/task_queue.hpp"
#include "ipc/queue_manager.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/base_service.hpp"
#include "services/reasoning_parser.hpp"
#include "services/streamable.hpp"
#include "services/tool_call_parser.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

using namespace tt::domain::llm;

class LLMService : public BaseService<LLMRequest, LLMResponse>,
                   public Streamable<LLMRequest, LLMStreamChunk> {
 public:
  using StreamCallback = std::function<void(const LLMStreamChunk&, bool)>;

  LLMService();

  LLMService(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
             std::unique_ptr<tt::worker::WorkerManager> workerManager,
             std::unique_ptr<ReasoningParser> reasoningParser,
             std::unique_ptr<IToolCallParser> toolCallParser,
             std::unique_ptr<tt::ipc::QueueManager> queueManager,
             size_t maxQueueSize = std::numeric_limits<size_t>::max());

  ~LLMService() override;

  LLMService(const LLMService&) = delete;
  LLMService& operator=(const LLMService&) = delete;

  void start() override;
  void stop() override;

  bool isModelReady() const override;

  void preProcess(LLMRequest& request) const override;

  void postProcess(LLMResponse& response) const override;

  void processStreamingRequest(
      LLMRequest request,
      std::function<void(LLMStreamChunk&, bool isFinal)> callback) override;

  void abortRequest(uint32_t taskId);

  ReasoningParser* getReasoningParser() const { return reasoningParser.get(); }

  tt::worker::WorkerManager* getWorkerManager() const {
    return workerManager.get();
  }

 protected:
  size_t currentQueueSize() const override;
  LLMResponse processRequest(LLMRequest request) override;

  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

  void streamingPostProcess(LLMStreamChunk&) const override {}

 private:
  struct StreamCallbackEntry {
    std::function<void(LLMStreamChunk&, bool)> callback;
    bool skip_special_tokens = true;
    // When true the consumer loop emits a chunk carrying only `token_id`
    // + `finish_reason` and skips decodeToken() / reasoning / tool-call
    // parsing entirely. Used by the Dynamo path because the wire-level
    // TokenChunk only forwards token_ids — the frontend handles
    // detokenization. Avoids loading tokenizer.json on the consumer
    // thread.
    bool skip_text_decode = false;
  };

  void startConsumers();
  void consumerLoopForWorker(size_t workerIdx);

  std::optional<StreamCallbackEntry> resolveCallback(uint32_t taskId,
                                                     bool isFinal);

  void init(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
            std::unique_ptr<tt::worker::WorkerManager> workerManager,
            std::unique_ptr<ReasoningParser> reasoningParser,
            std::unique_ptr<IToolCallParser> toolCallParser,
            std::unique_ptr<tt::ipc::QueueManager> queueManager,
            size_t maxQueueSize);

  std::vector<std::thread> consumerThreads;

  utils::ConcurrentMap<uint32_t, StreamCallbackEntry> streamCallbacks;
  mutable utils::ConcurrentMap<uint32_t, tt::domain::tool_calls::ToolChoice>
      toolChoiceMap;
  utils::ConcurrentMap<uint32_t, bool> reasoningSuppressedMap;

  std::atomic<size_t> pendingTasks{0};
  std::atomic<bool> running{false};

  std::shared_ptr<tt::ipc::ITaskQueue> taskQueue;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  std::unique_ptr<tt::ipc::QueueManager> queueManager;
  std::unordered_set<int64_t> stopTokenSet;
  std::unique_ptr<ReasoningParser> reasoningParser;
  std::unique_ptr<IToolCallParser> toolCallParser;
  std::unique_ptr<IToolCallParser> jsonToolCallParser;
};

}  // namespace tt::services
