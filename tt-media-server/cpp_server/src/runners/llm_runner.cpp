#include "runners/llm_runner.hpp"

#include <cassert>
#include <thread>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"

namespace tt::runners {
using namespace llm_engine;
using Config = tt::config::LLMConfig;

LLMRunner::LLMRunner(const Config& config,
                     ipc::TokenRingBuffer<65536>* resultQueue,
                     ITaskQueue* taskQueue)
    : config_(config), result_queue_(resultQueue) {
  scheduler_ =
      makeScheduler(config_, taskQueue, tt::config::maxInFlightCount());

  auto decodeCb = [this](const TokenResult& result) {
    ZoneScopedN("LLMRunner::process_token_result");
    Sequence* seq = scheduler_->findSequence(result.taskId);

    assert(seq);

    if (result.isError) {
      scheduler_->removeSequence(result.taskId);
      auto shared = ipc::SharedToken{
          .token_index = 0,
          .flags = static_cast<uint32_t>(ipc::SharedToken::FLAG_FINAL |
                                         ipc::SharedToken::FLAG_ERROR),
          .token_id = 0,
          .task_id = result.taskId.id,
          .padding = {},
      };
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
          .task_id = result.taskId.id,
          .padding = {},
      };
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

LLMRunner::~LLMRunner() { exit(); }

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

void LLMRunner::step() {
  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) return;
  ZoneScopedN("LLMRunner::step");
  model_runner_->run(seqs, is_prefill);
}
}  // namespace tt::runners
