#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/engine/debug.hpp"

#include <algorithm>

namespace llm_engine {

LLMEngine::LLMEngine(const Config& config) : config_(config) {
  LLM_ENGINE_LOG("llm_engine") << "construct" << std::endl;
  auto decode_cb = [this](const DecodeResult& result) {
    decode_queue_.push(result);
  };
  model_runner_ = make_model_runner(config_, std::move(decode_cb));
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

StepResult LLMEngine::step() {
  drain_decode_results();

  StepResult result;
  collect_finished(result);

  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) {
    result.num_tokens = 0;
    return result;
  }

  std::vector<int64_t> token_ids = model_runner_->run(seqs, is_prefill);

  if (is_prefill) {
    scheduler_->postprocess(seqs, token_ids);
    collect_finished(result);
  }
  // For decode, run() returns immediately. Results arrive via the reader
  // thread into decode_queue_ and will be drained at the top of a future
  // step() call.

  if (is_prefill) {
    int total = 0;
    for (Sequence* s : seqs) {
      total += static_cast<int>(s->size());
    }
    result.num_tokens = total;
  } else {
    result.num_tokens = -static_cast<int>(seqs.size());
  }
  LLM_ENGINE_LOG("llm_engine") << "step " << (is_prefill ? "prefill" : "decode")
                             << " n=" << seqs.size() << " num_tokens=" << result.num_tokens << std::endl;

  return result;
}

void LLMEngine::drain_decode_results() {
  for (const auto& dr : decode_queue_.drain()) {
    auto it = std::find_if(sequences_.begin(), sequences_.end(),
                           [&](const auto& seq) { return seq->seq_id == dr.seq_id; });
    if (it == sequences_.end()) continue;

    Sequence* seq = it->get();
    if (seq->status_ != SequenceStatus::IN_FLIGHT) continue;

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> token_ids = {dr.token_id};
    scheduler_->postprocess(seqs, token_ids);

    LLM_ENGINE_LOG("llm_engine") << "drain seq_id=" << dr.seq_id
                               << " token_id=" << dr.token_id << std::endl;
  }
}

void LLMEngine::collect_finished(StepResult& result) {
  for (const auto& seq_ptr : sequences_) {
    if (seq_ptr->is_finished() && reported_seq_ids_.insert(seq_ptr->seq_id).second) {
      result.outputs.emplace_back(seq_ptr->seq_id, seq_ptr->completion_token_ids());
      LLM_ENGINE_LOG("llm_engine") << "finished seq_id=" << seq_ptr->seq_id
                                 << " completion_tokens=" << seq_ptr->num_completion_tokens() << std::endl;
    }
  }
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
  while (!is_finished()) {
    StepResult r = step();
    for (auto& [seq_id, tokens] : r.outputs) {
      outputs.push_back({seq_id, std::move(tokens)});
    }
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
