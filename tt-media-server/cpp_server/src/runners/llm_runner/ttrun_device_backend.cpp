#include "runners/llm_runner/device_backend.hpp"
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
#include <sys/wait.h>
#include <unistd.h>

namespace llm_engine {

namespace {

class TtRunDeviceBackend : public IDeviceBackend {
 public:
  explicit TtRunDeviceBackend(const Config&)
      : write_shm_("/tt_ipc_c2p_"), read_shm_("/tt_ipc_p2c_") {
    write_shm_.open();
    read_shm_.open();
  }

  ~TtRunDeviceBackend() override = default;

  void init() override {
    const char* home = std::getenv("TT_METAL_HOME");
    if (!home || !*home) {
      throw std::runtime_error("TtRunDeviceBackend: TT_METAL_HOME not set");
    }
    std::string tt_metal_home(home);
    std::string ttrun_py = tt_metal_home + "/ttnn/ttnn/distributed/ttrun.py";
    std::string rank_binding = tt_metal_home + "/bh_4x2_multi_mesh_rank_binding.yaml";
    std::string hello_script = "/home/user/tt-inference-server/tt-media-server/cpp_server/src/runners/runner.py";
    std::string python_path = tt_metal_home + "/python_env/bin/python";

    std::vector<std::string> args = {
        python_path, ttrun_py, "--rank-binding", rank_binding, hello_script};
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
      throw std::runtime_error("TtRunDeviceBackend: fork failed");
    }
    if (pid == 0) {
      setpgid(0, 0);  // become process group leader so killpg covers all descendants
      setenv("TT_IPC_SHM_C2P", write_shm_.getName().c_str(), 1);
      setenv("TT_IPC_SHM_P2C", read_shm_.getName().c_str(), 1);
      setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
      execv(args[0].c_str(), argv.data());
      _exit(127);
    }
    setpgid(pid, pid);  // also set from parent side to avoid the race before child runs
    child_pid_ = pid;
    std::cout << "TtRunDeviceBackend: started tt-run pid " << pid << std::endl;
  }

  void write(const Sequence& seq) override {
    size_t& index = token_index_per_task_[seq.task_id];
    if (index >= kFixedReplySequence.size()) {
      index = 0;
    }
    int64_t token_id = kFixedReplySequence[index++];
    write_shm_.write(seq.task_id.id, static_cast<uint64_t>(token_id));
  }

  bool read(TokenResult* result) override {
    std::vector<uint8_t> bytes;
    while (!stop_.load(std::memory_order_relaxed)) {
      if (read_shm_.try_read(bytes)) {
        result->task_id = TaskID::deserialize(
            reinterpret_cast<const char*>(bytes.data()), 36);
        std::memcpy(&result->token_id, bytes.data() + 36, sizeof(result->token_id));
        return true;
      }
    }
    return false;
  }

  void terminate() override {
    stop_.store(true);
    if (child_pid_ > 0) {
      // Kill the python launcher and its direct process group.
      killpg(child_pid_, SIGTERM);
      // prterun/mpirun spawns MPI ranks in their own process group; kill both
      // the orchestrator and the python rank processes (cmdline: runner.py).
      std::system("pkill -TERM -f prterun");
      std::system("pkill -TERM -f runner.py");
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      // Force-kill anything that survived the grace period.
      killpg(child_pid_, SIGKILL);
      std::system("pkill -KILL -f prterun");
      std::system("pkill -KILL -f runner.py");
      waitpid(child_pid_, nullptr, 0);
      child_pid_ = -1;
    }
  }

 private:
  SharedMemory write_shm_;
  SharedMemory read_shm_;
  std::atomic<bool> stop_{false};
  pid_t child_pid_ = -1;
  std::unordered_map<TaskID, size_t> token_index_per_task_;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_ttrun(const Config& config) {
  return std::make_unique<TtRunDeviceBackend>(config);
}

}  // namespace llm_engine
