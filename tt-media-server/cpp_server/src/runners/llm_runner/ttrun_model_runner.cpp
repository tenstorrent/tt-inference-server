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

struct TtRunConfig {
  TtRunConfig() {
    const char* home = std::getenv("TT_METAL_HOME");
    if (!home || !*home) {
      throw std::runtime_error("TtRunModelRunner: TT_METAL_HOME not set");
    }
    tt_metal_home = std::string(home);
    ttrun_py = tt_metal_home + "ttnn/ttnn/distributed/ttrun.py";
    rank_binding = tt_metal_home + "bh_4x2_multi_mesh_rank_binding.yaml";

    // Check if we should use mock runner (no tt-run needed)
    const char* use_mock = std::getenv("TT_USE_MOCK_RUNNER");
    if (use_mock && std::string(use_mock) == "1") {
      script = "/localdev/idjuric/tt-inference-server/tt-media-server/cpp_server/src/runners/mock_runner.py";
      use_ttrun = false;
    } else {
      script = "/localdev/idjuric/tt-inference-server/tt-media-server/cpp_server/src/runners/runner.py";
      use_ttrun = true;
    }

    python_path = tt_metal_home + "python_env/bin/python";

    // Use environment variables if set, otherwise use defaults
    const char* c2p = std::getenv("TT_IPC_SHM_C2P");
    const char* p2c = std::getenv("TT_IPC_SHM_P2C");
    write_shm_name = c2p ? std::string(c2p) : "tt_ipc_c2p";
    read_shm_name = p2c ? std::string(p2c) : "tt_ipc_p2c";
  }

  std::string write_shm_name;
  std::string read_shm_name;
  std::string ttrun_py;
  std::string rank_binding;
  std::string script;
  std::string python_path;
  std::string tt_metal_home;
  bool use_ttrun = false;
};

class TtRunModelRunner : public IModelRunner {
 public:
  TtRunModelRunner(const Config& config, DecodeCallback callback)
      : config_(config),
        decode_callback_(std::move(callback)),
        ttrun_config_(),
        device_input_(ttrun_config_.write_shm_name),
        device_output_(ttrun_config_.read_shm_name) {
    LLM_ENGINE_LOG("model_runner:ttrun") << "Using shared memory: C2P="
                                         << ttrun_config_.write_shm_name
                                         << " P2C=" << ttrun_config_.read_shm_name << std::endl;
    device_input_.open();
    device_output_.open();
    reader_thread_ = std::thread([this] { reader_loop(); });
  }

  ~TtRunModelRunner() override { exit(); }

  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override {
    LLM_ENGINE_LOG("model_runner:ttrun") << (is_prefill ? "prefill" : "decode")
                                         << " batch_size=" << seqs.size() << std::endl;
    if (is_prefill) {
      for (Sequence* seq : seqs) {
        device_input_.write(seq->task_id.id, seq->token_ids_, seq->sampling_params->max_tokens);
      }
    } else {
      // Do nothing - device manager on device will just loop tokens on its own
      // for (Sequence* seq : seqs) {
      //   size_t& index = token_index_for_sequence_[seq->task_id];
      //   if (index >= kFixedReplySequence.size()) {
      //     index = 0;
      //   }
      //   int64_t token_id = kFixedReplySequence[index++];
      //   device_input_.write(seq->task_id.id, static_cast<uint64_t>(token_id));
      // }
    }
  }

  void exit() override {
    if (stop_.exchange(true)) return;
    terminate_child();
    if (reader_thread_.joinable()) reader_thread_.join();
    LLM_ENGINE_LOG("model_runner:ttrun") << "exit" << std::endl;
  }

 private:
  void ttrun() {
    if (ttrun_config_.use_ttrun) {
      // Launch with tt-run for multi-rank execution
      char* argv[] = {
          ttrun_config_.python_path.data(),
          ttrun_config_.ttrun_py.data(),
          const_cast<char*>("--rank-binding"),
          ttrun_config_.rank_binding.data(),
          ttrun_config_.script.data(),
          nullptr};

      pid_t pid = fork();
      if (pid < 0) {
        throw std::runtime_error("TtRunModelRunner: fork failed");
      }
      if (pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        setenv("TT_IPC_SHM_C2P", device_input_.getName().c_str(), 1);
        setenv("TT_IPC_SHM_P2C", device_output_.getName().c_str(), 1);
        setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
        execv(argv[0], argv);
        _exit(127);
      }
      child_pid_ = pid;
      LLM_ENGINE_LOG("model_runner:ttrun") << "started tt-run pid " << pid << std::endl;
    } else {
      // Launch mock runner directly (no tt-run)
      char* argv[] = {
          ttrun_config_.python_path.data(),
          ttrun_config_.script.data(),
          nullptr};

      pid_t pid = fork();
      if (pid < 0) {
        throw std::runtime_error("TtRunModelRunner: fork failed");
      }
      if (pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        setenv("TT_IPC_SHM_C2P", device_input_.getName().c_str(), 1);
        setenv("TT_IPC_SHM_P2C", device_output_.getName().c_str(), 1);
        execv(argv[0], argv);
        _exit(127);
      }
      child_pid_ = pid;
      LLM_ENGINE_LOG("model_runner:ttrun") << "started mock runner pid " << pid << std::endl;
    }
  }

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
        TokenResult result;
        result.task_id = TaskID::deserialize(
            reinterpret_cast<const char*>(payload), TaskID::kSerializedSize);

        // Extract first token_id if available (next 8 bytes after task_id)
        if (num_token_ids > 0 && payload_size >= TaskID::kSerializedSize + 8) {
          int64_t token_id;
          std::memcpy(&token_id, payload + TaskID::kSerializedSize, sizeof(int64_t));
          result.token_id = token_id;
          LLM_ENGINE_LOG("model_runner:ttrun") << "Decoded token: task_id=" << result.task_id.id
                                               << " token_id=" << token_id << std::endl;
        } else {
          result.token_id = 0;
          LLM_ENGINE_LOG("model_runner:ttrun") << "No token_id in message" << std::endl;
        }

        decode_callback_(result);
      } else {
        std::this_thread::yield();
      }
    }
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop exited" << std::endl;
  }  void terminate_child() {
    if (child_pid_ > 0) {
      kill(child_pid_, SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      kill(child_pid_, SIGKILL);
      waitpid(child_pid_, nullptr, WNOHANG);
      child_pid_ = -1;
    }
  }

  Config config_;
  DecodeCallback decode_callback_;
  TtRunConfig ttrun_config_;
  SharedMemory device_input_;
  SharedMemory device_output_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
  pid_t child_pid_ = -1;
  std::unordered_map<TaskID, size_t> token_index_for_sequence_;
};

}  // namespace

std::unique_ptr<IModelRunner> make_ttrun_model_runner(const Config& config,
                                                      DecodeCallback callback) {
  return std::make_unique<TtRunModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
