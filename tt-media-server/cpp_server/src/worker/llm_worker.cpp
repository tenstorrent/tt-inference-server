#include "worker/llm_worker.hpp"

namespace tt::worker {

LLMWorker::LLMWorker(WorkerConfig& cfg, const llm_engine::Config& llm_engine_config): BaseWorker(cfg), llm_engine_config_(llm_engine_config) {
    on_token_ = [this](llm_engine::TaskID task_id, uint64_t token_id, bool finished) {
            auto token = ipc::SharedToken{
                .token_index = 0,
                .flags = static_cast<uint32_t>(finished ? 1 : 0),
                .token_id = token_id,
                .task_id = {},
                .padding = {},
            };
            std::strncpy(token.task_id, task_id.id.c_str(), sizeof(token.task_id) - 1);
            token.task_id[sizeof(token.task_id) - 1] = '\0';
            result_queue->push(token);
        };
    is_ready = true;
}

LLMWorker::~LLMWorker() {
    stop();
}

void LLMWorker::start() {
    for (const auto& [key, value] : cfg_.env_vars) {
        setenv(key.c_str(), value.c_str(), 1);
    }

    auto scheduler = std::make_unique<llm_engine::Scheduler>(
        llm_engine_config_, cfg_.task_queue.get()
    );
    llm_engine_ = std::make_unique<llm_engine::LLMEngine>(
        llm_engine_config_,
        on_token_,
        std::move(scheduler)
    );
    llm_engine_->run();
}

void LLMWorker::stop() {
    if (llm_engine_) {
        llm_engine_->stop();
    }
}

}