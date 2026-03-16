#include "worker/single_process_worker.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "utils/runner_factory.hpp"
#include <csignal>
#include <sys/wait.h>
#include <iostream>
#include <chrono>
#include <thread>

namespace tt::worker {

SingleProcessWorker::SingleProcessWorker(WorkerConfig& cfg)
    : cfg(std::move(cfg)) {

    pid = getpid();
    worker_id = cfg.worker_id;
    is_ready = true;
}

SingleProcessWorker::~SingleProcessWorker() = default;

void SingleProcessWorker::start() {
    tracy_config::TracySetThreadName(
        ("Worker-" + to_string(cfg.worker_id)).c_str());

    for (const auto& [key, value] : cfg.env_vars) {
        setenv(key.c_str(), value.c_str(), 1);
    }

    {
        ZoneScopedN("Worker::init");
        runner_ = tt::utils::runner_factory::create_runner(
            tt::config::model_service(),
            cfg.runner_config,
            cfg.result_queue.get(),
            cfg.task_queue.get(),
            cfg.cancel_queue.get()
        );
    }
    runner_->start();
}

void SingleProcessWorker::stop() {
    ZoneScopedN("Worker::stop");
    if (runner_) {
        runner_->stop();
    }
    if (pid > 0) {
        killpg(pid, SIGTERM);

        int status;
        int wait_result = waitpid(pid, &status, WNOHANG);
        if (wait_result == 0) {
            this_thread::sleep_for(chrono::milliseconds(500));
            wait_result = waitpid(pid, &status, WNOHANG);
            if (wait_result == 0) {
                killpg(pid, SIGKILL);
                waitpid(pid, &status, 0);
            }
        }
        cout << "[SingleProcessWorker] Worker " << worker_id << " exited\n" << flush;
    }
}

} // namespace tt::worker
