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
 * Process-wide registry mapping (ModelService, ModelRunnerType) pairs to
 * runner factories. Mirrors Python's `tt_model_runners.runner_fabric.AVAILABLE
 * _RUNNERS` dict, but keyed on a 2-tuple so the same modality can offer
 * multiple runners (e.g. LLM has MOCK / LLAMA / PIPELINE_MANAGER / PREFILL).
 *
 * `runner_factory::createRunner` delegates to this registry; modules add
 * runners by calling `registerRunner(...)` from
 * `services::registerBuiltinModalities()` (or, in the future, from a per-
 * modality static initializer).
 *
 * Lookup falls back from `(service, type)` to `(service, MOCK)` and then to
 * the first registered factory for the service, so that callers passing
 * unsupported runner types still get a sensible default — matching the
 * pre-refactor behaviour of the switch statement.
 */
class RunnerRegistry {
 public:
  using RunnerFactory = std::function<std::unique_ptr<runners::IRunner>(
      const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
      ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue)>;

  RunnerRegistry(const RunnerRegistry&) = delete;
  RunnerRegistry& operator=(const RunnerRegistry&) = delete;

  static RunnerRegistry& instance();

  /** Register a factory for a (service, runner type) pair. */
  void registerRunner(config::ModelService service,
                      config::ModelRunnerType type, RunnerFactory factory);

  /** Construct the runner. Returns nullptr if no factory matches. */
  std::unique_ptr<runners::IRunner> create(
      config::ModelService service, config::ModelRunnerType type,
      const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
      ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue) const;

  /** True iff a factory is registered for the (service, type) pair. */
  bool has(config::ModelService service, config::ModelRunnerType type) const;

  /** Remove all registrations. Test-only helper. */
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
