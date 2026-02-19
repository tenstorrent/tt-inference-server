#include "runners/llm_engine/engine/llm_engine.hpp"
#include "worker/base_worker.hpp"
#include "config/settings.hpp"
#include <functional>

namespace tt::worker {

class LLMWorker: public BaseWorker {
public:
    LLMWorker(
        WorkerConfig& cfg, const llm_engine::Config& llm_engine_config = tt::config::llm_engine_config()
    );
    ~LLMWorker() override;

    void start() override;
    void stop() override;
    
private:
    std::unique_ptr<llm_engine::LLMEngine> llm_engine_;
    std::function<void(llm_engine::TaskID task_id, uint64_t token_id, bool finished)> on_token_;
    llm_engine::Config llm_engine_config_;
};


}