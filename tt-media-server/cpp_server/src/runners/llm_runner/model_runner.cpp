#include "runners/llm_runner/model_runner.hpp"

#include <stdexcept>

#ifdef USE_METAL_CPP_LIB
#include "runners/h2h_kv_cache_migrator.hpp"
#include "runners/llama_model_runner.hpp"
#endif

#include "config/settings.hpp"

namespace llm_engine {

using Config = tt::config::LLMConfig;
using ModelRunnerType = tt::config::ModelRunnerType;

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback);
#ifdef USE_METAL_CPP_LIB
std::unique_ptr<IModelRunner> makeLlamaModelRunner(
    const Config& config, DecodeCallback callback,
    std::unique_ptr<IKVCacheMigrator> migrator);
#endif

#ifdef USE_METAL_CPP_LIB
namespace {
std::unique_ptr<IKVCacheMigrator> createKVCacheMigrator() {
  auto mode = tt::config::llmMode();
  if (mode == tt::config::LLMMode::REGULAR) return nullptr;

  auto host = tt::config::kvMigrationHost();
  auto port = tt::config::kvMigrationPort();

  // Decode server listens; prefill client connects to it.
  auto socketMode = (mode == tt::config::LLMMode::DECODE_ONLY)
                        ? H2HKVCacheMigrator::Mode::SERVER
                        : H2HKVCacheMigrator::Mode::CLIENT;

  return std::make_unique<H2HKVCacheMigrator>(socketMode, host, port);
}
}  // namespace
#endif

std::unique_ptr<IModelRunner> makeModelRunner(const Config& config,
                                              DecodeCallback callback) {
  switch (config.runner_type) {
    case ModelRunnerType::MOCK:
      return makeMockModelRunner(config, std::move(callback));
#ifdef USE_METAL_CPP_LIB
    case ModelRunnerType::LLAMA:
      return makeLlamaModelRunner(config, std::move(callback),
                                  createKVCacheMigrator());
#endif
    default:
      throw std::invalid_argument("Invalid model runner type");
  }
}

}  // namespace llm_engine
