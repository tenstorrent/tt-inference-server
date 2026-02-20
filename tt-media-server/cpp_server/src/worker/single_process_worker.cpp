#include "worker/single_process_worker.hpp"
#include "profiling/tracy.hpp"
#include "utils/runner_factory.hpp"
#include <csignal>
#include <sys/wait.h>
#include <iostream>
#include <chrono>
#include <thread>

namespace tt::worker {

SingleProcessWorker::SingleProcessWorker(WorkerConfig& cfg, const llm_engine::Config& llm_engine_config)
    : cfg(move(cfg)), llm_engine_config_(llm_engine_config) {
    
    pid = getpid();
    worker_id = cfg.worker_id;
    
    on_token_ = [this](llm_engine::TaskID task_id, uint64_t token_id, bool finished, bool is_stop_token) {
        uint32_t flags = 0;
        if (finished) flags |= ipc::SharedToken::FLAG_FINAL;
        if (is_stop_token) flags |= ipc::SharedToken::FLAG_STOP_TOKEN;
        auto token = ipc::SharedToken{
            .token_index = 0,
            .flags = flags,
            .token_id = token_id,
            .task_id = {},
            .padding = {},
        };
        strncpy(token.task_id, task_id.id.c_str(), sizeof(token.task_id) - 1);
        token.task_id[sizeof(token.task_id) - 1] = '\0';
        this->cfg.result_queue->push(token);
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
            llm_engine_config_,
            on_token_,
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
