// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/runner_registry.hpp"

#include "utils/logger.hpp"

namespace tt::utils {

RunnerRegistry& RunnerRegistry::instance() {
  static RunnerRegistry registry;
  return registry;
}

void RunnerRegistry::registerRunner(config::ModelService service,
                                    config::ModelRunnerType type,
                                    RunnerFactory factory) {
  factories_[{service, type}] = std::move(factory);
}

std::unique_ptr<runners::IRunner> RunnerRegistry::create(
    config::ModelService service, config::ModelRunnerType type,
    const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
    ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue) const {
  auto exact = factories_.find({service, type});
  if (exact != factories_.end() && exact->second) {
    return exact->second(config, resultQueue, taskQueue, cancelQueue);
  }

  // MOCK is the convention safety net. No "first available" scan: unordered
  // map iteration is unordered, which would make selection non-deterministic.
  auto mock = factories_.find({service, config::ModelRunnerType::MOCK});
  if (mock != factories_.end() && mock->second) {
    TT_LOG_WARN(
        "[RunnerRegistry] No factory registered for ({}, {}); falling back to "
        "MOCK",
        config::toString(service), config::toString(type));
    return mock->second(config, resultQueue, taskQueue, cancelQueue);
  }

  return nullptr;
}

bool RunnerRegistry::has(config::ModelService service,
                         config::ModelRunnerType type) const {
  auto it = factories_.find({service, type});
  return it != factories_.end() && static_cast<bool>(it->second);
}

void RunnerRegistry::clear() { factories_.clear(); }

}  // namespace tt::utils
