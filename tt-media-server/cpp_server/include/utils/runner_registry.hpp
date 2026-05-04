// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/result_queue.hpp"
#include "ipc/task_queue.hpp"
#include "runners/runner_interface.hpp"

namespace tt::utils {

/**
 * Maps `(ModelService, ModelRunnerType)` to runner factories. Services
 * register themselves from `services::registerBuiltinModelServices()` and
 * `runner_factory::createRunner` delegates here.
 *
 * Lookup falls back from `(service, type)` -> `(service, MOCK)` -> first
 * factory registered for `service`, logging a warning when it does so.
 */
class RunnerRegistry {
 public:
  using RunnerFactory = std::function<std::unique_ptr<runners::IRunner>(
      const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
      ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue)>;

  RunnerRegistry(const RunnerRegistry&) = delete;
  RunnerRegistry& operator=(const RunnerRegistry&) = delete;

  static RunnerRegistry& instance();

  void registerRunner(config::ModelService service,
                      config::ModelRunnerType type, RunnerFactory factory);

  /** Returns nullptr if neither an exact match nor any fallback matches. */
  std::unique_ptr<runners::IRunner> create(
      config::ModelService service, config::ModelRunnerType type,
      const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
      ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue) const;

  bool has(config::ModelService service, config::ModelRunnerType type) const;

  /** Test-only. */
  void clear();

 private:
  RunnerRegistry() = default;

  struct KeyHash {
    size_t operator()(
        const std::pair<config::ModelService, config::ModelRunnerType>& k)
        const noexcept {
      return (static_cast<size_t>(k.first) << 16) ^
             static_cast<size_t>(k.second);
    }
  };

  std::unordered_map<std::pair<config::ModelService, config::ModelRunnerType>,
                     RunnerFactory, KeyHash>
      factories_;
};

}  // namespace tt::utils
