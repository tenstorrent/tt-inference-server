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

// Each variant arm exposes `runner_type`, so std::visit picks the right field
// uniformly. New modality configs only need to declare a `runner_type` member.
config::ModelRunnerType runnerTypeFromConfig(
    const config::RunnerConfig& config) {
  return std::visit([](const auto& cfg) { return cfg.runner_type; }, config);
}

}  // namespace

std::unique_ptr<runners::IRunner> createRunner(
    config::ModelService service, const config::RunnerConfig& config,
    ipc::IResultQueue* resultQueue, tt::ipc::ITaskQueue* taskQueue,
    ipc::ICancelQueue* cancelQueue) {
  // Idempotent; required for callers that bypass service_factory (e.g. tests).
  services::registerBuiltinModelServices();

  const config::ModelRunnerType runnerType = runnerTypeFromConfig(config);

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
