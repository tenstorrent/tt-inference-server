#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <unistd.h>
#include "runners/runner_interface.hpp"
#include "runners/runner_config.hpp"
#include "runners/runner_result.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "ipc/shared_memory.hpp"

namespace tt::worker {

using namespace std;

struct WorkerConfig {
    unordered_map<string, string> env_vars;
    shared_ptr<llm_engine::ITaskQueue> task_queue;
    shared_ptr<tt::ipc::TokenRingBuffer<65536>> result_queue;
    int worker_id;
    tt::runners::RunnerConfig runner_config;
};

/**
 * Single process worker that runs an LLM engine.
 * Handles task processing and token generation.
 */
class SingleProcessWorker {
public:
    SingleProcessWorker(WorkerConfig& cfg);
    ~SingleProcessWorker();

    void start();
    void stop();

    // Public member variables for compatibility with existing LLMService
    pid_t pid{-1};
    bool is_ready{false};
    bool is_alive{true};
    int worker_id{-1};
    WorkerConfig cfg;

private:
    unique_ptr<tt::runners::IRunner> runner_;
    tt::runners::ResultCallback on_result_;
};

} // namespace tt::worker
