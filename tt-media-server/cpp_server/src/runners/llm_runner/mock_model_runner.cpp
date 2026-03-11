#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"

namespace llm_engine {

namespace {

constexpr int64_t kWhitespaceTokenId = 223;

class MockModelRunner : public IModelRunner {
 public:
  MockModelRunner(const Config& config, DecodeCallback callback)
      : config_(config), decode_callback_(std::move(callback)) {}

  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override {
    LLM_ENGINE_LOG("model_runner:mock") << (is_prefill ? "prefill" : "decode")
                                        << " batch_size=" << seqs.size() << std::endl;
    if (is_prefill) {
      for (Sequence* seq : seqs) {
        decode_callback_(TokenResult(seq->task_id, kWhitespaceTokenId));
      }
    } else {
      for (Sequence* seq : seqs) {
        decode_callback_(TokenResult(seq->task_id, static_cast<uint64_t>(seq->last_token + 1)));
      }
    }
  }

  void exit() override {
    LLM_ENGINE_LOG("model_runner:mock") << "exit" << std::endl;
  }

 private:
  Config config_;
  DecodeCallback decode_callback_;
};

}  // namespace

std::unique_ptr<IModelRunner> make_mock_model_runner(const Config& config,
                                                     DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
