#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"
#include "profiling/tracy.hpp"

#include <cstdlib>
#include <thread>

namespace llm_engine {

namespace {

constexpr int64_t kWhitespaceTokenId = 223;

class MockModelRunner : public IModelRunner {
 public:
  MockModelRunner(const Config& config, DecodeCallback callback)
      : config_(config), decode_callback_(std::move(callback)) {}

  static int token_delay_us() {
    const char* s = std::getenv("MOCK_TOKEN_DELAY_US");
    return s ? std::atoi(s) : 0;
  }

  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override {
    ZoneScopedN("MockModelRunner::run");
    LLM_ENGINE_LOG("model_runner:mock") << (is_prefill ? "prefill" : "decode")
                                        << " batch_size=" << seqs.size() << std::endl;
    const int delay = token_delay_us();
    if (is_prefill) {
      ZoneScopedN("MockModelRunner::prefill");
      for (Sequence* seq : seqs) {
        if (delay > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(delay));
        }
        decode_callback_(TokenResult(seq->task_id, kWhitespaceTokenId));
      }
    } else {
      ZoneScopedN("MockModelRunner::decode");
      for (Sequence* seq : seqs) {
        if (delay > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(delay));
        }
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
