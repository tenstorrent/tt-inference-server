#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/deepseek_runner/shared_memory.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>

namespace llm_engine {

namespace {

struct ShmNames {
  ShmNames() {
    const char* c2p = std::getenv("TT_IPC_SHM_C2P");
    const char* p2c = std::getenv("TT_IPC_SHM_P2C");
    write = c2p ? std::string(c2p) : "/tt_ipc_c2p";
    read = p2c ? std::string(p2c) : "/tt_ipc_p2c";
  }

  std::string write;
  std::string read;
};

class DeepSeekModelRunner : public IModelRunner {
 public:
  DeepSeekModelRunner(const Config& config, DecodeCallback callback)
      : config_(config),
        decode_callback_(std::move(callback)),
        shm_names_(),
        device_input_(shm_names_.write),
        device_output_(shm_names_.read) {
    LLM_ENGINE_LOG("model_runner:deepseek") << "Using shared memory: C2P="
                                            << shm_names_.write
                                            << " P2C=" << shm_names_.read << std::endl;
    device_input_.open();
    device_output_.open();
    reader_thread_ = std::thread([this] { reader_loop(); });
  }

  ~DeepSeekModelRunner() override { exit(); }

  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override {
    if (!is_prefill) return;
    for (Sequence* seq : seqs) {
      LLM_ENGINE_LOG("model_runner:deepseek") << "Writing to device: task_id=" << seq->task_id.id
                                              << " num_tokens=" << seq->token_ids_.size()
                                              << " max_tokens=" << seq->sampling_params->max_tokens
                                              << std::endl;
      device_input_.write(seq->task_id.id, seq->token_ids_, seq->sampling_params->max_tokens);
    }
  }

  void exit() override {
    if (stop_.exchange(true)) return;
    if (reader_thread_.joinable()) reader_thread_.join();
    LLM_ENGINE_LOG("model_runner:deepseek") << "exit" << std::endl;
  }

 private:
  void reader_loop() {
    LLM_ENGINE_LOG("model_runner:deepseek") << "Reader loop started" << std::endl;
    ReadResult buf;
    while (!stop_.load(std::memory_order_relaxed)) {
      if (device_output_.try_read(buf)) {
        TokenResult result;
        result.token_id = buf.token;
        result.task_id = TaskID::deserialize(buf.task_id, TaskID::kSerializedSize);
        decode_callback_(result);
      } else {
        std::this_thread::yield();
      }
    }
    LLM_ENGINE_LOG("model_runner:deepseek") << "Reader loop exited" << std::endl;
  }

  Config config_;
  DecodeCallback decode_callback_;
  ShmNames shm_names_;
  SharedMemory device_input_;
  SharedMemory device_output_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
};

}  // namespace

std::unique_ptr<IModelRunner> make_deepseek_model_runner(const Config& config,
                                                         DecodeCallback callback) {
  return std::make_unique<DeepSeekModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
