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
            [&](const runners::TokenPayload& tok) {
                auto shared = ipc::SharedToken{
                    .token_index = 0,
                    .flags = static_cast<uint32_t>(tok.finished ? 1 : 0),
                    .token_id = tok.token_id,
                    .task_id = {},
                    .padding = {},
                };
                strncpy(shared.task_id, result.task_id.c_str(), sizeof(shared.task_id) - 1);
                shared.task_id[sizeof(shared.task_id) - 1] = '\0';
                this->cfg.result_queue->push(shared);
            },
            [&](const runners::EmbeddingPayload&) {
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
