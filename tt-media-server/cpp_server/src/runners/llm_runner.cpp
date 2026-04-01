#include "runners/llm_runner.hpp"

#include <cassert>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/paged_memory_manager.hpp"

namespace tt::runners {
using namespace llm_engine;
using Config = tt::config::LLMConfig;

LLMRunner::LLMRunner(const Config& config,
                     ipc::TokenRingBuffer<65536>* resultQueue,
                     ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue)
    : config_(config), result_queue_(resultQueue), cancel_queue_(cancelQueue) {
  scheduler_ =
      makeScheduler(config_, taskQueue, tt::config::maxInFlightCount());

  if (tt::config::llmMode() != config::LLMMode::PREFILL_ONLY) {
    memoryManager = std::make_unique<services::PagedMemoryManager>(
        scheduler_->blockManager());
    memoryThread = std::thread([this] { memoryLoop(); });
  }

  auto decodeCb = [this](const TokenResult& result) {
    ZoneScopedN("LLMRunner::process_token_result");
    Sequence* seq = scheduler_->findSequence(result.taskId);

    // Sequence was aborted between model run and token callback — discard.
    if (!seq || seq->isAborted()) return;

    if (result.isError) {
      scheduler_->removeSequence(result.taskId);
      auto shared = ipc::SharedToken{
          .token_index = 0,
          .flags = static_cast<uint32_t>(ipc::SharedToken::FLAG_FINAL |
                                         ipc::SharedToken::FLAG_ERROR),
          .token_id = 0,
          .task_id = {},
          .padding = {},
      };
      strncpy(shared.task_id, result.taskId.id.c_str(),
              sizeof(shared.task_id) - 1);
      shared.task_id[sizeof(shared.task_id) - 1] = '\0';
      while (!result_queue_->push(shared)) {
        std::this_thread::yield();
      }
      return;
    }

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> tokenIds = {static_cast<int64_t>(result.tokenId)};
    scheduler_->postprocess(seqs, tokenIds);

    bool finished = seq->isFinished();

    {
      ZoneScopedN("ResultQueue::push");
      auto shared = ipc::SharedToken{
          .token_index = 0,
          .flags = static_cast<uint32_t>(finished ? ipc::SharedToken::FLAG_FINAL
                                                  : 0),
          .token_id = result.tokenId,
          .task_id = {},
          .padding = {},
      };
      strncpy(shared.task_id, result.taskId.id.c_str(),
              sizeof(shared.task_id) - 1);
      shared.task_id[sizeof(shared.task_id) - 1] = '\0';
      while (!result_queue_->push(shared)) {
        std::this_thread::yield();
      }
    }

    if (finished) {
      scheduler_->removeSequence(result.taskId);
    }
  };

  model_runner_ = makeModelRunner(config_, std::move(decodeCb));
}

LLMRunner::~LLMRunner() {
  stop();
  if (memoryThread.joinable()) {
    memoryThread.join();
  }
  exit();
}

void LLMRunner::exit() {
  if (model_runner_) {
    model_runner_->exit();
  }
}

void LLMRunner::run() {
  while (!stopped_.load(std::memory_order_relaxed)) {
    step();
  }
}

void LLMRunner::stop() { stopped_.store(true, std::memory_order_relaxed); }

void LLMRunner::memoryLoop() {
  while (!stopped_.load(std::memory_order_relaxed)) {
    auto task = memoryManager->getRequest();
    if (task) {
      memoryManager->handleRequest(*task);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void LLMRunner::step() {
  // Drain cancel queue before scheduling so aborted sequences are excluded
  // from the next batch.
  if (cancel_queue_) {
    std::vector<TaskID> cancelled;
    cancel_queue_->tryPopAll(cancelled);
    for (const auto& taskId : cancelled) {
      scheduler_->abortRequest(taskId);
    }
  }

  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) return;
  ZoneScopedN("LLMRunner::step");
  model_runner_->run(seqs, is_prefill);
}
}  // namespace tt::runners
