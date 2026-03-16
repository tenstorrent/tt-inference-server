#include "runners/llm_runner.hpp"

#include <cassert>
#include <thread>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"

namespace tt::runners {
using namespace llm_engine;
using Config = tt::config::LLMConfig;

LLMRunner::LLMRunner(const Config& config,
                     ipc::TokenRingBuffer<65536>* result_queue,
                     ITaskQueue* task_queue)
    : config_(config), result_queue_(result_queue) {
  scheduler_ =
      make_scheduler(config_, task_queue, tt::config::max_in_flight_count());

  auto decode_cb = [this](const TokenResult& result) {
    ZoneScopedN("LLMRunner::process_token_result");
    Sequence* seq = scheduler_->find_sequence(result.task_id);
    assert(seq);

    if (result.is_error) {
      scheduler_->removeSequence(result.task_id);
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
      while (!result_queue_->push(shared)) {
        std::this_thread::yield();
      }
      return;
    }

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> token_ids = {static_cast<int64_t>(result.token_id)};
    scheduler_->postprocess(seqs, token_ids);

    bool finished = seq->is_finished();

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
      while (!result_queue_->push(shared)) {
        std::this_thread::yield();
      }
    }

    if (finished) {
      scheduler_->removeSequence(result.task_id);
    }
  };

  model_runner_ = make_model_runner(config_, std::move(decode_cb));
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
