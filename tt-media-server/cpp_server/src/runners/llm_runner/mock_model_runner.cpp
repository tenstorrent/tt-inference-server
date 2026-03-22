#include "profiling/tracy.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace llm_engine {

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
        decodeCallback(TokenResult(seq->taskId, K_WHITESPACE_TOKEN_ID));
      }
    } else {
      ZoneScopedN("MockModelRunner::decode");
      for (Sequence* seq : seqs) {
        decodeCallback(TokenResult(seq->taskId,
                                   static_cast<uint64_t>(seq->lastToken + 1)));
      }
    }
  }

  void exit() override {
    LLM_ENGINE_LOG("model_runner:mock") << "exit" << std::endl;
  }

 private:
  Config config;
  DecodeCallback decodeCallback;
};

}  // namespace

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
