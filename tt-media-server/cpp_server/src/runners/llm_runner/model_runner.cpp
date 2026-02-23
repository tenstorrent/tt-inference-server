#include "runners/llm_runner/model_runner.hpp"
#include "runners/llm_runner/debug.hpp"
#include "runners/llm_runner/device_backend.hpp"
#include "runners/llm_runner/sequence.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace llm_engine {

namespace {

void run_ttrun_hello_world_thread(std::string tt_metal_home) {
  std::string ttrun_py = tt_metal_home + "/ttnn/ttnn/distributed/ttrun.py";
  std::string rank_binding = tt_metal_home + "/bh_4x2_multi_mesh_rank_binding.yaml";
  std::string hello_script = tt_metal_home + "/ttnn/ttnn/distributed/ttrun_hello_world.py";
  std::string python_path = tt_metal_home + "/python_env/bin/python";

  std::vector<std::string> args = {
      python_path, ttrun_py, "--rank-binding", rank_binding,hello_script};
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);

  pid_t pid = fork();
  std::cout << "ttrun hello world: forked process with pid " << pid << std::endl;
  if (pid < 0) {
    LLM_ENGINE_LOG("model_runner") << "ttrun hello world: fork failed" << std::endl;
    return;
  }
  if (pid == 0) {
    execv(args[0].c_str(), argv.data());
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
    LLM_ENGINE_LOG("model_runner") << "ttrun hello world: exec failed (127), is python3 in PATH?" << std::endl;
  }
}

void launch_ttrun_hello_world_nonblocking() {
  const char* home = std::getenv("TT_METAL_HOME");
  if (!home || !*home) {
    std::cout << "ttrun hello world: TT_METAL_HOME not set" << std::endl;
    return;
  };
  std::string tt_metal_home(home);
  std::thread([tt_metal_home]() { run_ttrun_hello_world_thread(tt_metal_home); }).detach();
}

}  // namespace

constexpr int64_t kWhitespaceTokenId = 223;

void DecodeQueue::push(const DecodeResult& result) {
  std::lock_guard lock(mutex_);
  pending_.push_back(result);
}

std::vector<DecodeResult> DecodeQueue::drain() {
  std::lock_guard lock(mutex_);
  std::vector<DecodeResult> out;
  out.swap(pending_);
  return out;
}

ModelRunnerStub::ModelRunnerStub(const Config& config, DecodeCallback callback,
                                std::unique_ptr<IDeviceBackend> backend)
    : config_(config),
      decode_callback_(std::move(callback)),
      backend_(std::move(backend)) {
  backend_->init();
  reader_thread_ = std::thread([this] { reader_loop(); });
}

ModelRunnerStub::~ModelRunnerStub() {
  exit();
}

void ModelRunnerStub::reader_loop() {
  DecodeResult result;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (!backend_->read(&result)) break;
    if (stop_.load(std::memory_order_relaxed)) break;
    decode_callback_(result);
  }
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    std::cout << "ttrun hello world: running" << std::endl;
    launch_ttrun_hello_world_nonblocking();
    for (Sequence* seq : seqs) {
      decode_callback_({seq->task_id, kWhitespaceTokenId});
    }
  } else {
    backend_->write(*seqs[0]);
  }
}

void ModelRunnerStub::exit() {
  if (stop_.exchange(true)) return;
  backend_->terminate();
  if (reader_thread_.joinable()) reader_thread_.join();
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  auto backend = make_device_backend(config);
  return std::make_unique<ModelRunnerStub>(config, std::move(callback), std::move(backend));
}

}  // namespace llm_engine
