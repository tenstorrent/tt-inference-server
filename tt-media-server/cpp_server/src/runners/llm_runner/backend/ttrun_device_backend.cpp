#include "runners/llm_runner/backend/device_backend.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/fixed_reply_sequence.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/shared_memory.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace llm_engine::backend {

struct TtRunDeviceBackendConfig {

  TtRunDeviceBackendConfig(){ // default constructor
      const char* home = std::getenv("TT_METAL_HOME");
        if (!home || !*home) {
          throw std::runtime_error("TtRunDeviceBackend: TT_METAL_HOME not set");
        }
      ttMetalHome = std::string(home);
      ttrunPy = ttMetalHome + "ttnn/ttnn/distributed/ttrun.py";
      rankBinding = ttMetalHome + "bh_4x2_multi_mesh_rank_binding.yaml";
      script = "/home/user/tt-inference-server/tt-media-server/cpp_server/src/runners/runner.py";
      pythonPath = ttMetalHome + "python_env/bin/python";
      writeShmName = "tt_ipc_c2p_";
      readShmName  = "tt_ipc_p2c_";
  }
  TtRunDeviceBackendConfig(
    const std::string& writeShmName,
    const std::string& readShmName,
    const std::string& ttrunPy,
    const std::string& rankBinding,
    const std::string& script,
    const std::string& pythonPath
  ):
    writeShmName(writeShmName),
    readShmName(readShmName),
    ttrunPy(ttrunPy),
    rankBinding(rankBinding),
    script(script),
    pythonPath(pythonPath) {}

  std::string writeShmName;
  std::string readShmName;
  std::string ttrunPy;
  std::string rankBinding;
  std::string script;
  std::string pythonPath;
  std::string ttMetalHome;
};

namespace {

class TtRunDeviceBackend : public IDeviceBackend {
 public:
  explicit TtRunDeviceBackend(const Config& config, const TtRunDeviceBackendConfig& ttrun_config):
    deviceInput(ttrun_config.writeShmName),
    deviceOutput(ttrun_config.readShmName),
    ttrunConfig(ttrun_config)
  {
    deviceInput.open();
    deviceOutput.open();
  }

  ~TtRunDeviceBackend() override = default;

  void init() override {
    char* argv[] = {
      ttrunConfig.pythonPath.data(),
      ttrunConfig.ttrunPy.data(),
      const_cast<char*>("--rank-binding"),
      ttrunConfig.rankBinding.data(),
      ttrunConfig.script.data(),
      nullptr
    };

    pid_t pid = fork();
    if (pid < 0) {
      throw std::runtime_error("TtRunDeviceBackend: fork failed");
    }
    if (pid == 0) {
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      setenv("TT_IPC_SHM_C2P", deviceInput.getName().c_str(), 1);
      setenv("TT_IPC_SHM_P2C", deviceOutput.getName().c_str(), 1);
      setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
      execv(argv[0], argv);
      _exit(127); // this should never happen
    }
    childPid = pid;
    std::cout << "TtRunDeviceBackend: started tt-run pid " << pid << std::endl;
  }

  void write(const std::vector<Sequence*>& seqs) override {
    for (Sequence* seq : seqs) {
      size_t& index = tokenIndexForSequence[seq->task_id];
      if (index >= kFixedReplySequence.size()) {
        index = 0;
      }
      int64_t token_id = kFixedReplySequence[index++];
      deviceInput.write(seq->task_id.id, static_cast<uint64_t>(token_id));
  }
  }

  bool read(TokenResult* result) override {
    std::vector<uint8_t> bytes;
    while (!stop.load(std::memory_order_relaxed)) {
      if (deviceOutput.try_read(bytes)) {
        result->task_id = TaskID::deserialize(
            reinterpret_cast<const char*>(bytes.data()), TaskID::kSerializedSize
        );
        std::memcpy(&result->token_id, bytes.data() + TaskID::kSerializedSize, sizeof(result->token_id));
        return true;
      }
      std::this_thread::yield();
    }
    return false;
  }

  void terminate() override {
    stop.store(true, std::memory_order_relaxed);
    if (childPid > 0) {
      kill(childPid, SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      kill(childPid, SIGKILL);
      waitpid(childPid, nullptr, WNOHANG);
      childPid = -1;
    }
  }

 private:
  SharedMemory deviceInput;
  SharedMemory deviceOutput;
  std::atomic<bool> stop{false};
  TtRunDeviceBackendConfig ttrunConfig;
  pid_t childPid = -1;
  std::unordered_map<TaskID, size_t> tokenIndexForSequence;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_ttrun(const Config& config) {
  return std::make_unique<TtRunDeviceBackend>(config, TtRunDeviceBackendConfig());
}

}  // namespace llm_engine::backend
