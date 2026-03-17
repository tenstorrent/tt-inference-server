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
    : config(config), resultQueue(resultQueue) {
  scheduler =
      makeScheduler(config, taskQueue, tt::config::maxInFlightCount());

  auto decodeCb = [this](const TokenResult& result) {
    ZoneScopedN("LLMRunner::process_token_result");
    Sequence* seq = scheduler->findSequence(result.task_id);

    assert(seq);

    if (result.is_error) {
      scheduler->removeSequence(result.task_id);
      auto shared = ipc::SharedToken{
          .token_index = 0,
          .flags = static_cast<uint32_t>(ipc::SharedToken::FLAG_FINAL |
                                         ipc::SharedToken::FLAG_ERROR),
          .token_id = 0,
          .task_id = {},
          .padding = {},
      };
      strncpy(shared.task_id, result.task_id.id.c_str(),
              sizeof(shared.task_id) - 1);
      shared.task_id[sizeof(shared.task_id) - 1] = '\0';
      while (!this->resultQueue->push(shared)) {
        std::this_thread::yield();
      }
      return;
    }

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> tokenIds = {static_cast<int64_t>(result.token_id)};
    scheduler->postprocess(seqs, tokenIds);

    bool finished = seq->isFinished();

    {
      ZoneScopedN("ResultQueue::push");
      auto shared = ipc::SharedToken{
          .token_index = 0,
          .flags = static_cast<uint32_t>(finished ? ipc::SharedToken::FLAG_FINAL
                                                  : 0),
          .token_id = result.token_id,
          .task_id = {},
          .padding = {},
      };
      strncpy(shared.task_id, result.task_id.id.c_str(),
              sizeof(shared.task_id) - 1);
      shared.task_id[sizeof(shared.task_id) - 1] = '\0';
      while (!this->resultQueue->push(shared)) {
        std::this_thread::yield();
      }
    }

    if (finished) {
      scheduler->removeSequence(result.task_id);
    }
  };

  modelRunner = make_model_runner(config, std::move(decodeCb));
}

LLMRunner::~LLMRunner() { exit(); }

void LLMRunner::exit() {
  if (modelRunner) {
    modelRunner->exit();
  }
}

void LLMRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    step();
  }
}

void LLMRunner::stop() { stopped.store(true, std::memory_order_relaxed); }

void LLMRunner::step() {
  auto [seqs, is_prefill] = scheduler->schedule();
  if (seqs.empty()) return;
  ZoneScopedN("LLMRunner::step");
  modelRunner->run(seqs, is_prefill);
}
}  // namespace tt::runners
