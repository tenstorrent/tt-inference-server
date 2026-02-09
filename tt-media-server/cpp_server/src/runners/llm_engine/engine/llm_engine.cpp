#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/engine/debug.hpp"
#include <algorithm>
#include <condition_variable>
#include <mutex>

namespace llm_engine {

LLMEngine::LLMEngine(const Config& config) : config_(config) {
  LLM_ENGINE_LOG("llm_engine") << "construct" << std::endl;
  model_runner_ = make_model_runner(config_);
  model_runner_->set_on_tokens_callback(
      [this](std::vector<TokenEntry> entries) {
        on_tokens_from_runner(std::move(entries));
      });
  scheduler_ = std::make_unique<Scheduler>(config_);
  if (config_.eos < 0) {
    config_.eos = 0;
  }
}

LLMEngine::~LLMEngine() {
  exit();
}

void LLMEngine::exit() {
  if (model_runner_) {
    LLM_ENGINE_LOG("llm_engine") << "exit" << std::endl;
    model_runner_->exit();
  }
}

void LLMEngine::add_request(std::vector<int64_t> prompt,
                             const SamplingParams& sampling_params) {
  auto seq = std::make_unique<Sequence>(std::move(prompt), sampling_params);
  Sequence* ptr = seq.get();
  sequences_.push_back(std::move(seq));
  scheduler_->add(*ptr);
  LLM_ENGINE_LOG("llm_engine") << "add_request seq_id=" << ptr->seq_id
                             << " prompt_len=" << ptr->size()
                             << " max_tokens=" << ptr->max_tokens << std::endl;
}

void LLMEngine::on_tokens_from_runner(std::vector<TokenEntry> entries) {
  std::vector<Sequence*> seqs;
  std::vector<int64_t> token_ids;
  seqs.reserve(entries.size());
  token_ids.reserve(entries.size());
  for (const auto& [token_id, user_id] : entries) {
    Sequence* seq = nullptr;
    for (auto& s : sequences_) {
      if (s->seq_id == user_id) {
        seq = s.get();
        break;
      }
    }
    if (seq) {
      seqs.push_back(seq);
      token_ids.push_back(token_id);
    }
  }
  if (seqs.empty()) return;

  scheduler_->postprocess(seqs, token_ids);

  StepResult result;
  if (current_is_prefill_) {
    int total = 0;
    for (Sequence* s : seqs) {
      total += static_cast<int>(s->size());
    }
    result.num_tokens = total;
  } else {
    result.num_tokens = -static_cast<int>(seqs.size());
  }
  LLM_ENGINE_LOG("llm_engine") << "step " << (current_is_prefill_ ? "prefill" : "decode")
                               << " n=" << seqs.size() << " num_tokens=" << result.num_tokens << std::endl;
  for (Sequence* seq : seqs) {
    if (seq->is_finished()) {
      result.outputs.emplace_back(seq->seq_id, seq->completion_token_ids());
      LLM_ENGINE_LOG("llm_engine") << "step seq_id=" << seq->seq_id << " finished"
                                  << " completion_tokens=" << seq->num_completion_tokens() << std::endl;
    }
  }
  current_step_done_(std::move(result));
}

void LLMEngine::step(StepResultCallback on_step_done) {
  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) {
    LLM_ENGINE_LOG("llm_engine") << "step empty batch" << std::endl;
    on_step_done(StepResult{});
    return;
  }

  current_step_done_ = std::move(on_step_done);
  current_is_prefill_ = is_prefill;
  model_runner_->run(seqs, is_prefill);
}

bool LLMEngine::is_finished() const {
  return scheduler_->is_finished();
}

std::vector<std::vector<int64_t>> LLMEngine::generate(
    const std::vector<std::vector<int64_t>>& prompts,
    const std::vector<SamplingParams>& sampling_params) {
  LLM_ENGINE_LOG("llm_engine") << "generate n_prompts=" << prompts.size() << std::endl;
  std::vector<SamplingParams> params = sampling_params;
  if (params.size() != prompts.size()) {
    params.assign(prompts.size(), SamplingParams());
  }
  for (size_t i = 0; i < prompts.size(); ++i) {
    add_request(prompts[i], params[i]);
  }

  std::vector<std::pair<int, std::vector<int64_t>>> outputs;
  std::mutex gen_mutex;
  std::condition_variable gen_cv;
  bool gen_done = false;

  auto on_step = std::make_shared<StepResultCallback>();
  *on_step = [this, &outputs, &gen_mutex, &gen_cv, &gen_done, on_step](StepResult r) {
    for (auto& o : r.outputs) {
      outputs.push_back(std::move(o));
    }
    if (!is_finished()) {
      step(*on_step);
    } else {
      std::lock_guard lock{gen_mutex};
      gen_done = true;
      gen_cv.notify_one();
    }
  };
  step(*on_step);

  {
    std::unique_lock lock{gen_mutex};
    gen_cv.wait(lock, [&] { return gen_done; });
  }

  LLM_ENGINE_LOG("llm_engine") << "generate done n_outputs=" << outputs.size() << std::endl;
  std::sort(outputs.begin(), outputs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<std::vector<int64_t>> result;
  result.reserve(outputs.size());
  for (auto& [_, tokens] : outputs) {
    result.push_back(std::move(tokens));
  }
  return result;
}

std::vector<std::vector<int64_t>> LLMEngine::generate(
    const std::vector<std::vector<int64_t>>& prompts,
    const std::vector<int>& max_tokens_per_request) {
  std::vector<SamplingParams> params;
  params.reserve(prompts.size());
  if (max_tokens_per_request.size() == prompts.size()) {
    for (int m : max_tokens_per_request) {
      SamplingParams sp;
      sp.max_tokens = m;
      params.push_back(sp);
    }
  }
  return generate(prompts, params);
}

}  // namespace llm_engine
