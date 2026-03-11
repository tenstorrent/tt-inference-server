#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/fixed_reply_sequence.hpp"
#include "runners/llm_runner/shared_memory.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace llm_engine {

namespace {

constexpr int64_t kWhitespaceTokenId = 223;

struct SharedMemoryConfig {
  SharedMemoryConfig() {
    // Use environment variables if set, otherwise use defaults
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
        sharemem_config_(),
        device_input_(sharemem_config_.write_shm_name),
        device_output_(sharemem_config_.read_shm_name) {
    LLM_ENGINE_LOG("model_runner:ttrun") << "Using shared memory: C2P="
                                         << sharemem_config_.write_shm_name
                                         << " P2C=" << sharemem_config_.read_shm_name << std::endl;
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
    std::vector<uint8_t> bytes;
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop started" << std::endl;
    while (!stop_.load(std::memory_order_relaxed)) {
      if (device_output_.try_read(bytes)) {
        LLM_ENGINE_LOG("model_runner:ttrun") << "Received message, size=" << bytes.size() << std::endl;

        // New format: max_tokens (4) + num_token_ids (4) + payload (task_id 36 + token_ids variable)
        if (bytes.size() < 8) {
          LLM_ENGINE_LOG("model_runner:ttrun") << "Invalid message size: " << bytes.size() << std::endl;
          continue;
        }

        uint32_t max_tokens;
        uint32_t num_token_ids;
        std::memcpy(&max_tokens, bytes.data(), 4);
        std::memcpy(&num_token_ids, bytes.data() + 4, 4);

        const uint8_t* payload = bytes.data() + 8;
        size_t payload_size = bytes.size() - 8;

        if (payload_size < TaskID::kSerializedSize) {
          LLM_ENGINE_LOG("model_runner:ttrun") << "Payload too small: " << payload_size << std::endl;
          continue;
        }

        // Extract task_id (first 36 bytes of payload)
        TaskID result_task_id = TaskID::ipc_deserialize(
            reinterpret_cast<const char*>(payload), TaskID::kSerializedSize);

        uint64_t result_token_id = 0;
        // Extract first token_id if available (next 8 bytes after task_id)
        if (num_token_ids > 0 && payload_size >= TaskID::kSerializedSize + 8) {
          int64_t token_id;
          std::memcpy(&token_id, payload + TaskID::kSerializedSize, sizeof(int64_t));
          result_token_id = static_cast<uint64_t>(token_id);
          LLM_ENGINE_LOG("model_runner:ttrun") << "Decoded token: task_id=" << result_task_id.id
                                               << " token_id=" << result_token_id << std::endl;
        } else {
          LLM_ENGINE_LOG("model_runner:ttrun") << "No token_id in message" << std::endl;
        }

        TokenResult result(result_task_id, result_token_id);
        decode_callback_(result);
      } else {
        std::this_thread::yield();
      }
    }
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop exited" << std::endl;
  }

  Config config_;
  DecodeCallback decode_callback_;
  SharedMemoryConfig sharemem_config_;
  SharedMemory device_input_;
  SharedMemory device_output_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
  std::unordered_map<TaskID, size_t> token_index_for_sequence_;
};

}  // namespace

std::unique_ptr<IModelRunner> make_ttrun_model_runner(const Config& config,
                                                      DecodeCallback callback) {
  return std::make_unique<TtRunModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
