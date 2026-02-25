#include "runners/llm_runner/device_backend.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/fixed_reply_sequence.hpp"
#include "runners/llm_runner/sequence.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif

namespace llm_engine {

namespace {

constexpr size_t kPageSize = 64;
constexpr uint32_t kNumSlots = 1024;

struct ShmChannel {
  uint32_t read_pos;
  uint32_t write_pos;
  char pad[56];
  char slots[kNumSlots][kPageSize];
};
static_assert(sizeof(ShmChannel) == 64 + kNumSlots * kPageSize);

struct ShmRegion {
  ShmChannel c2p;
  ShmChannel p2c;
};

static void token_to_page(const TaskID& task_id, int64_t token_id, char* page) {
  auto id_bytes = task_id.serialize();
  std::memcpy(page, id_bytes.data(), std::min(id_bytes.size(), TaskID::kSerializedSize));
  std::memset(page + TaskID::kSerializedSize, 0, kPageSize - TaskID::kSerializedSize);
  std::memcpy(page + TaskID::kSerializedSize, &token_id, sizeof(token_id));
}

static void page_to_result(const char* page, TokenResult* result) {
  result->task_id = TaskID::deserialize(page, TaskID::kSerializedSize);
  std::memcpy(&result->token_id, page + TaskID::kSerializedSize, sizeof(result->token_id));
}

class TtRunDeviceBackend : public IDeviceBackend {
 public:
  explicit TtRunDeviceBackend(const Config&) {
    shm_name_ = "tt_ipc_" + std::to_string(getpid());
    std::string shm_path = "/" + shm_name_;
    int fd = shm_open(shm_path.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
      throw std::runtime_error("TtRunDeviceBackend: shm_open failed");
    }
    if (ftruncate(fd, sizeof(ShmRegion)) != 0) {
      close(fd);
      shm_unlink(shm_path.c_str());
      throw std::runtime_error("TtRunDeviceBackend: ftruncate failed");
    }
    region_ = static_cast<ShmRegion*>(
        mmap(nullptr, sizeof(ShmRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    if (region_ == MAP_FAILED) {
      shm_unlink(shm_path.c_str());
      throw std::runtime_error("TtRunDeviceBackend: mmap failed");
    }
    std::memset(region_, 0, sizeof(ShmRegion));
  }

  ~TtRunDeviceBackend() override {
    if (region_ && region_ != MAP_FAILED) {
      munmap(region_, sizeof(ShmRegion));
    }
    if (!shm_name_.empty()) {
      shm_unlink(("/" + shm_name_).c_str());
    }
  }

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
      setenv("TT_IPC_SHM", shm_name_.c_str(), 1);
      setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
      execv(args[0].c_str(), argv.data());
      _exit(127);
    }
    child_pid_ = pid;
    std::cout << "TtRunDeviceBackend: started tt-run pid " << pid << std::endl;
  }

  void write(const Sequence& seq) override {
    std::cout << "TtRunDeviceBackend: write start" << std::endl;
    size_t& index = token_index_per_task_[seq.task_id];
    if (index >= kFixedReplySequence.size()) {
      index = 0;
    }
    int64_t token_id = kFixedReplySequence[index++];
    ShmChannel& ch = region_->c2p;
    uint32_t w = __atomic_load_n(&ch.write_pos, __ATOMIC_RELAXED);
    while (w - __atomic_load_n(&ch.read_pos, __ATOMIC_ACQUIRE) >= kNumSlots) {
      CPU_RELAX();
    }
    char* slot = ch.slots[w % kNumSlots];
    token_to_page(seq.task_id, token_id, slot);
    __atomic_store_n(&ch.write_pos, w + 1, __ATOMIC_RELEASE);
    std::cout << "TtRunDeviceBackend: write done" << std::endl;
  }

  bool read(TokenResult* result) override {
    std::cout << "TtRunDeviceBackend: read start" << std::endl;
    ShmChannel& ch = region_->p2c;
    while (!stop_.load(std::memory_order_relaxed)) {
      uint32_t r = __atomic_load_n(&ch.read_pos, __ATOMIC_RELAXED);
      uint32_t w = __atomic_load_n(&ch.write_pos, __ATOMIC_ACQUIRE);
      if (r != w) {
        const char* slot = ch.slots[r % kNumSlots];
        page_to_result(slot, result);
        __atomic_store_n(&ch.read_pos, r + 1, __ATOMIC_RELEASE);
        std::cout << "TtRunDeviceBackend: read done (r=" << r << " w=" << w << ")"
                  << std::endl;
        return true;
      }
      CPU_RELAX();
    }
    return false;
  }

  void terminate() override {
    stop_.store(true);
    auto command = "pkill -SIGINT prterun";
    std::system(command);
    if (child_pid_ > 0) {
      kill(child_pid_, SIGTERM);
      waitpid(child_pid_, nullptr, 0);
      child_pid_ = -1;
    }
  }

 private:
  std::string shm_name_;
  ShmRegion* region_ = nullptr;
  std::atomic<bool> stop_{false};
  pid_t child_pid_ = -1;
  std::unordered_map<TaskID, size_t> token_index_per_task_;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_ttrun(const Config& config) {
  return std::make_unique<TtRunDeviceBackend>(config);
}

}  // namespace llm_engine
