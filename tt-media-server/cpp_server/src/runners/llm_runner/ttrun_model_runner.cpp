#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/shared_memory.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace llm_engine {

namespace {

constexpr int64_t kWhitespaceTokenId = 223;

struct ShmNames {
  ShmNames() {
    const char* c2p = std::getenv("TT_IPC_SHM_C2P");
    const char* p2c = std::getenv("TT_IPC_SHM_P2C");
    write_shm_name = c2p ? std::string(c2p) : "tt_ipc_c2p";
    read_shm_name = p2c ? std::string(p2c) : "tt_ipc_p2c";
  }

  std::string write_shm_name;
  std::string read_shm_name;
};

class TtRunModelRunner : public IModelRunner {
 public:
  TtRunModelRunner(const Config& config, DecodeCallback callback)
      : config_(config),
        decode_callback_(std::move(callback)),
        shm_names_(),
        device_input_(shm_names_.write_shm_name),
        device_output_(shm_names_.read_shm_name) {
    LLM_ENGINE_LOG("model_runner:ttrun") << "Using shared memory: C2P="
                                         << shm_names_.write_shm_name
                                         << " P2C=" << shm_names_.read_shm_name << std::endl;
    device_input_.open();
    device_output_.open();
    reader_thread_ = std::thread([this] { reader_loop(); });
  }

  ~TtRunModelRunner() override { exit(); }

  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override {
    LLM_ENGINE_LOG("model_runner:ttrun") << (is_prefill ? "prefill" : "decode")
                                         << " batch_size=" << seqs.size() << std::endl;
    if (is_prefill) {
      // we write for prefill only, device will loop decode tokens on its own
      for (Sequence* seq : seqs) {
        LLM_ENGINE_LOG("model_runner:ttrun") << "Writing to device: task_id=" << seq->task_id.id
                                         << " num_tokens=" << seq->token_ids_.size()
                                         << " max_tokens=" << seq->sampling_params->max_tokens
                                         << std::endl;
        device_input_.write(seq->task_id.id, seq->token_ids_, seq->sampling_params->max_tokens);
      }
    }
  }

  void exit() override {
    if (stop_.exchange(true)) return;
    if (reader_thread_.joinable()) reader_thread_.join();
    LLM_ENGINE_LOG("model_runner:ttrun") << "exit" << std::endl;
  }

 private:
  void reader_loop() {
    ReadResult read_buf;
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop started" << std::endl;
    while (!stop_.load(std::memory_order_relaxed)) {
      if (device_output_.try_read(read_buf)) {
        TokenResult result;
        result.task_id = TaskID::deserialize(
            read_buf.task_id.data(), TaskID::kSerializedSize);
        result.token_id = read_buf.token_ids.empty() ? 0 : read_buf.token_ids[0];

        LLM_ENGINE_LOG("model_runner:ttrun") << "Decoded token: task_id=" << result.task_id.id
                                             << " token_id=" << result.token_id << std::endl;
        decode_callback_(result);
      } else {
        std::this_thread::yield();
      }
    }
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop exited" << std::endl;
  }

  Config config_;
  DecodeCallback decode_callback_;
  ShmNames shm_names_;
  PrefillSharedMemory device_input_;
  DecodeSharedMemory device_output_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
};

}  // namespace

std::unique_ptr<IModelRunner> make_ttrun_model_runner(const Config& config,
                                                      DecodeCallback callback) {
  return std::make_unique<TtRunModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
