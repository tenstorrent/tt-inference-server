#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "runners/llm_engine/engine/task_queue.hpp"
#include "ipc/shared_memory.hpp"

namespace tt::worker {
    
struct WorkerConfig {
    std::unordered_map<std::string, std::string> env_vars;
    std::shared_ptr<llm_engine::ITaskQueue> task_queue;
    std::shared_ptr<tt::ipc::TokenRingBuffer<65536>> result_queue;
    int worker_id;
};

class BaseWorker {
public:
    BaseWorker(WorkerConfig& cfg): result_queue(cfg.result_queue), cfg_(std::move(cfg)), task_queue(cfg.task_queue) {
        pid = getpid();
        worker_id = cfg_.worker_id;
    }
    virtual ~BaseWorker() = default;

    virtual void start() = 0;
    virtual void stop() = 0;
    pid_t pid{-1};
    bool is_ready{false};
    bool is_alive{true};
    std::shared_ptr<tt::ipc::TokenRingBuffer<65536>> result_queue;
    int worker_id{-1};
    std::shared_ptr<llm_engine::ITaskQueue> task_queue;
protected:
    WorkerConfig cfg_;
};

}