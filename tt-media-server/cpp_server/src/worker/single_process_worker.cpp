#include "worker/single_process_worker.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <thread>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/runner_factory.hpp"

namespace tt::worker {

namespace {

volatile sig_atomic_t gWorkerId =
    -1;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void fatalSignalHandler(int sig) {
  const char prefix[] = "[SingleProcessWorker] Worker ";
  const char mid[] = " killed by signal: ";
  const char suffix[] = "\n";

  char idBuf[16];
  int wid = gWorkerId;
  if (wid < 0) wid = 0;
  int len = 0;
  char tmp[16];
  int n = wid;
  if (n == 0) {
    tmp[len++] = '0';
  } else {
    while (n > 0) {
      tmp[len++] = '0' + (n % 10);
      n /= 10;
    }
  }
  for (int i = 0; i < len; ++i) idBuf[i] = tmp[len - 1 - i];

  const char* sigName = strsignal(sig);
  if (!sigName) sigName = "unknown";

  write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
  write(STDERR_FILENO, idBuf, len);
  write(STDERR_FILENO, mid, sizeof(mid) - 1);
  write(STDERR_FILENO, sigName, strlen(sigName));
  write(STDERR_FILENO, suffix, sizeof(suffix) - 1);

  signal(sig, SIG_DFL);
  raise(sig);
}

void installFatalSignalHandlers() {
  struct sigaction sa{};
  sa.sa_handler = fatalSignalHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
}

}  // namespace

SingleProcessWorker::SingleProcessWorker(WorkerConfig& cfg)
    : cfg(std::move(cfg)) {
  pid = getpid();
  worker_id = cfg.worker_id;
  is_ready = true;
}

SingleProcessWorker::~SingleProcessWorker() = default;

void SingleProcessWorker::start() {
  gWorkerId = worker_id;
  installFatalSignalHandlers();

  tracy_config::tracySetThreadName(
      ("Worker-" + std::to_string(cfg.worker_id)).c_str());

  for (const auto& [key, value] : cfg.env_vars) {
    setenv(key.c_str(), value.c_str(), 1);
  }

  {
    ZoneScopedN("Worker::init");
    runner_ = tt::utils::runner_factory::createRunner(
        tt::config::modelService(), cfg.runner_config, cfg.result_queue.get(),
        cfg.task_queue.get(), cfg.cancel_queue.get());
  }
  TT_LOG_INFO(
      "[SingleProcessWorker] Worker {} starting runner (warmup may take a "
      "while)",
      worker_id);
  try {
    runner_->start([this]() {
      try {
        tt::ipc::BoostIpcWarmupSignalQueue warmupQueue(
            tt::ipc::WARMUP_SIGNALS_QUEUE_NAME);
        warmupQueue.sendReady(worker_id);
        TT_LOG_INFO("[SingleProcessWorker] Worker {} signaled warmup complete",
                    worker_id);
      } catch (const std::exception& e) {
        TT_LOG_ERROR(
            "[SingleProcessWorker] Worker {} failed to signal warmup: {}",
            worker_id, e.what());
      }
    });
    TT_LOG_CRITICAL(
        "[SingleProcessWorker] Worker {} runner loop returned unexpectedly "
        "(runner type: {})",
        worker_id, runner_ ? runner_->runnerType() : "unknown");
  } catch (const std::exception& e) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorker] Worker {} CRASHED with exception: {} "
        "(runner type: {})",
        worker_id, e.what(), runner_ ? runner_->runnerType() : "unknown");
    throw;
  } catch (...) {
    TT_LOG_CRITICAL(
        "[SingleProcessWorker] Worker {} CRASHED with unknown exception "
        "(runner type: {})",
        worker_id, runner_ ? runner_->runnerType() : "unknown");
    throw;
  }
}

void SingleProcessWorker::stop() {
  ZoneScopedN("Worker::stop");
  if (runner_) {
    runner_->stop();
  }
  if (pid > 0) {
    killpg(pid, SIGTERM);

    int status;
    int waitResult = waitpid(pid, &status, WNOHANG);
    if (waitResult == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(
          tt::config::defaults::WORKER_STOP_TIMEOUT_MS));
      waitResult = waitpid(pid, &status, WNOHANG);
      if (waitResult == 0) {
        killpg(pid, SIGKILL);
        waitpid(pid, &status, 0);
      }
    }
    TT_LOG_INFO("[SingleProcessWorker] Worker {} exited", worker_id);
  }
}

}  // namespace tt::worker
