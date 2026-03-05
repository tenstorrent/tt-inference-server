#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/shared_memory.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
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
    writeSharedMemoryName = c2p ? std::string(c2p) : "/tt_ipc_c2p";
    readSharedMemoryName = p2c ? std::string(p2c) : "/tt_ipc_p2c";
  }

  std::string writeSharedMemoryName;
  std::string readSharedMemoryName;
};

class TtRunModelRunner : public IModelRunner {
 public:
  TtRunModelRunner(const Config& config, DecodeCallback callback)
      : config(config),
        decodeCallback(std::move(callback)),
        sharedMemoryConfig(),
        deviceInput(sharedMemoryConfig.writeSharedMemoryName),
        deviceOutput(sharedMemoryConfig.readSharedMemoryName) {
    LLM_ENGINE_LOG("model_runner:ttrun") << "Using shared memory: C2P="
                                         << sharedMemoryConfig.writeSharedMemoryName
                                         << " P2C=" << sharedMemoryConfig.readSharedMemoryName << std::endl;
    deviceInput.open();
    deviceOutput.open();
    readerThread = std::thread([this] { readerLoop(); });
  }

  ~TtRunModelRunner() override { exit(); }

  void run(const std::vector<Sequence*>& seqs, bool isPrefill) override {
    if (isPrefill) {
      // we write for prefill only, device will loop decode tokens on its own
      for (Sequence* seq : seqs) {
        LLM_ENGINE_LOG("model_runner:ttrun") << "Writing to device: task_id=" << seq->task_id.id
                                         << " num_tokens=" << seq->token_ids_.size()
                                         << " max_tokens=" << seq->sampling_params->max_tokens
                                         << std::endl;
        deviceInput.write(seq->task_id.id, seq->token_ids_, seq->sampling_params->max_tokens);
      }
    }
  }

  void exit() override {
    if (stop_.exchange(true)) return;
    if (readerThread.joinable()) readerThread.join();
    LLM_ENGINE_LOG("model_runner:ttrun") << "exit" << std::endl;
  }

 private:
  void readerLoop() {
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop started" << std::endl;
    ReadResult buf;
    while (!stop_.load(std::memory_order_relaxed)) {
      if (deviceOutput.try_read(buf)) {
        TokenResult result;
        result.token_id = buf.token;
        result.task_id = TaskID::deserialize(buf.task_id, TaskID::kSerializedSize);
        decodeCallback(result);
      } else {
        std::this_thread::yield();
      }
    }
    LLM_ENGINE_LOG("model_runner:ttrun") << "Reader loop exited" << std::endl;
  }

  Config config;
  DecodeCallback decodeCallback;
  SharedMemoryConfig sharedMemoryConfig;
  SharedMemory deviceInput;
  SharedMemory deviceOutput;
  std::atomic<bool> stop_{false};
  std::thread readerThread;
  std::unordered_map<TaskID, size_t> tokenIndexInSequence;
};

}  // namespace

std::unique_ptr<IModelRunner> make_ttrun_model_runner(const Config& config,
                                                      DecodeCallback callback) {
  return std::make_unique<TtRunModelRunner>(config, std::move(callback));
}

}  // namespace llm_engine
