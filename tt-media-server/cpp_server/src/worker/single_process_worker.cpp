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
    : cfg(move(cfg)) {

    pid = getpid();
    worker_id = cfg.worker_id;

    on_result_ = [this](const runners::RunnerResult& result) {
        std::visit(runners::overloaded{
            [&](const ipc::SharedToken& token) {
                this->cfg.result_queue->push(token);
            },
            [&](const ipc::SharedEmbedding&) {
                // TODO: Embedding IPC transport
            },
        }, result.payload);
    };
    is_ready = true;
}

SingleProcessWorker::~SingleProcessWorker() {
    stop();
}

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
            on_result_,
            cfg.task_queue.get()
        );
    }
    runner_->run();
}

void SingleProcessWorker::stop() {
    if (runner_) {
        runner_->stop();
    }
    if (pid > 0) {
        kill(pid, SIGTERM);

        int status;
        int wait_result = waitpid(pid, &status, WNOHANG);
        if (wait_result == 0) {
            this_thread::sleep_for(chrono::milliseconds(100));
            wait_result = waitpid(pid, &status, WNOHANG);
            if (wait_result == 0) {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
            }
        }
        cout << "[SingleProcessWorker] Worker " << worker_id << " exited\n" << flush;
    }
}

} // namespace tt::worker
