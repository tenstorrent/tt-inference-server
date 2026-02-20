#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <unistd.h>
#include "runners/runner_interface.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "ipc/shared_memory.hpp"
#include "config/settings.hpp"

namespace tt::worker {

using namespace std;

struct WorkerConfig {
    unordered_map<string, string> env_vars;
    shared_ptr<llm_engine::ITaskQueue> task_queue;
    shared_ptr<tt::ipc::TokenRingBuffer<65536>> result_queue;
    int worker_id;
};

/**
 * Single process worker that runs an LLM engine.
 * Handles task processing and token generation.
 */
class SingleProcessWorker {
public:
    SingleProcessWorker(
        WorkerConfig& cfg, 
        const llm_engine::Config& llm_engine_config = tt::config::llm_engine_config()
    );
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
    function<void(llm_engine::TaskID task_id, uint64_t token_id, bool finished, bool is_stop_token)> on_token_;
    llm_engine::Config llm_engine_config_;
};

} // namespace tt::worker
