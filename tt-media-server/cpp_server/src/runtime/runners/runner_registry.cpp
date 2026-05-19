// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/runner_registry.hpp"

#include "utils/logger.hpp"

namespace tt::utils {

RunnerRegistry& RunnerRegistry::instance() {
  static RunnerRegistry registry;
  return registry;
}

void RunnerRegistry::registerIpcRunner(config::ModelService service,
                                       config::ModelRunnerType type,
                                       IpcFactory factory) {
  ipc_factories_[{service, type}] = std::move(factory);
}

std::unique_ptr<runners::IRunner> RunnerRegistry::createIpc(
    config::ModelService service, config::ModelRunnerType type,
    const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
    ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue) const {
  auto exact = ipc_factories_.find({service, type});
  if (exact != ipc_factories_.end() && exact->second) {
    return exact->second(config, resultQueue, taskQueue, cancelQueue);
  }

  // Fall back to MOCK rather than scanning (unordered_map iteration is
  // non-deterministic).
  auto mock = ipc_factories_.find({service, config::ModelRunnerType::MOCK});
  if (mock != ipc_factories_.end() && mock->second) {
    TT_LOG_WARN(
        "[RunnerRegistry] No factory registered for ({}, {}); falling back to "
        "MOCK",
        config::toString(service), config::toString(type));
    return mock->second(config, resultQueue, taskQueue, cancelQueue);
  }

  return nullptr;
}

bool RunnerRegistry::hasIpc(config::ModelService service,
                            config::ModelRunnerType type) const {
  auto it = ipc_factories_.find({service, type});
  return it != ipc_factories_.end() && static_cast<bool>(it->second);
}

void RunnerRegistry::registerMediaRunner(config::ModelService service,
                                         config::ModelRunnerType type,
                                         MediaFactory factory) {
  media_factories_[{service, type}] = std::move(factory);
}

std::unique_ptr<runners::IRunnerBase> RunnerRegistry::createMediaBase(
    config::ModelService service, config::ModelRunnerType type,
    const config::RunnerConfig& config) const {
  auto exact = media_factories_.find({service, type});
  if (exact != media_factories_.end() && exact->second) {
    return exact->second(config);
  }
  return nullptr;
}

bool RunnerRegistry::hasMedia(config::ModelService service,
                              config::ModelRunnerType type) const {
  auto it = media_factories_.find({service, type});
  return it != media_factories_.end() && static_cast<bool>(it->second);
}

void RunnerRegistry::clear() {
  ipc_factories_.clear();
  media_factories_.clear();
}

}  // namespace tt::utils
