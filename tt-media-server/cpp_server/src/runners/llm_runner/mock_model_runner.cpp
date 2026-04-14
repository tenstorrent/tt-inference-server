#include "profiling/tracy.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;

namespace {

constexpr int64_t K_WHITESPACE_TOKEN_ID = 223;

class MockModelRunner : public IModelRunner {
 public:
  MockModelRunner(const Config& config, DecodeCallback callback)
      : config(config), decodeCallback(std::move(callback)) {}

  void run(const std::vector<Sequence*>& seqs, bool isPrefill) override {
    ZoneScopedN("MockModelRunner::run");
    LLM_ENGINE_LOG("model_runner:mock")
        << (isPrefill ? "prefill" : "decode")
        << " max_in_flight_count=" << seqs.size() << std::endl;
    if (isPrefill) {
      ZoneScopedN("MockModelRunner::prefill");
      for (Sequence* seq : seqs) {
        uint64_t tokenId = pickToken(seq, K_WHITESPACE_TOKEN_ID);
        decodeCallback(TokenResult(seq->taskId, tokenId));
      }
    } else {
      ZoneScopedN("MockModelRunner::decode");
      for (Sequence* seq : seqs) {
        uint64_t defaultToken = static_cast<uint64_t>(seq->getLastToken() + 1);
        uint64_t tokenId = pickToken(seq, defaultToken);
        decodeCallback(TokenResult(seq->taskId, tokenId));
      }
    }
  }

  void exit() override {
    LLM_ENGINE_LOG("model_runner:mock") << "exit" << std::endl;
  }

 private:
  static uint64_t pickToken(const Sequence* seq, uint64_t defaultToken) {
    const auto& allowed = seq->getSamplingParams().allowed_token_ids;
    if (!allowed.has_value() || allowed->empty()) return defaultToken;

    int target = static_cast<int>(defaultToken);
    for (int id : *allowed) {
      if (id == target) return defaultToken;
    }
    return static_cast<uint64_t>(allowed->front());
  }

  Config config;
  DecodeCallback decodeCallback;
};

}  // namespace

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace tt::runners::llm_engine
