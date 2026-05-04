// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/runner_registry.hpp"

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
  // 1) Exact match.
  auto exact = factories_.find({service, type});
  if (exact != factories_.end() && exact->second) {
    return exact->second(config, resultQueue, taskQueue, cancelQueue);
  }

  // 2) Fall back to MOCK for the same service.
  auto mock = factories_.find({service, config::ModelRunnerType::MOCK});
  if (mock != factories_.end() && mock->second) {
    return mock->second(config, resultQueue, taskQueue, cancelQueue);
  }

  // 3) Fall back to any factory registered for the service.
  for (const auto& [key, factory] : factories_) {
    if (key.first == service && factory) {
      return factory(config, resultQueue, taskQueue, cancelQueue);
    }
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
