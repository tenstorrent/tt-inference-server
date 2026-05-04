// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/runner_factory.hpp"

#include <stdexcept>
#include <variant>

#include "services/model_service_registration.hpp"
#include "utils/logger.hpp"
#include "utils/runner_registry.hpp"

namespace tt::utils::runner_factory {

namespace {

config::ModelRunnerType runnerTypeFromConfig(
    config::ModelService service, const config::RunnerConfig& config) {
  if (service == config::ModelService::LLM) {
    if (auto* llm = std::get_if<config::LLMConfig>(&config)) {
      return llm->runner_type;
    }
  }
  return config::ModelRunnerType::MOCK;
}

}  // namespace

std::unique_ptr<runners::IRunner> createRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue) {
  // Idempotent; required for callers that bypass service_factory (e.g. tests).
  services::registerBuiltinModelServices();

  const config::ModelRunnerType runnerType =
      runnerTypeFromConfig(service, config);

  auto runner = RunnerRegistry::instance().create(
      service, runnerType, config, resultQueue, taskQueue, cancelQueue);
  if (!runner) {
    TT_LOG_ERROR(
        "[RunnerFactory] No runner registered for service+type; refusing to "
        "create");
    throw std::runtime_error("No runner registered for the requested service");
  }
  return runner;
}

}  // namespace tt::utils::runner_factory
