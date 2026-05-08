// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/result_queue.hpp"
#include "ipc/task_queue.hpp"
#include "runners/runner_base.hpp"
#include "runners/runner_interface.hpp"

namespace tt::utils {

/**
 * Single registry for both runner families, keyed on (ModelService,
 * ModelRunnerType).
 *
 *  - IPC factories build loop-driven runners (LLM, embedding) inside the
 *    worker process; they need queue plumbing.
 *  - Media factories build direct-call runners (image; audio, TTS, video
 *    next) inside the parent process from config alone.
 *
 * Both families share lookup, fallback, and clear() semantics so adding a
 * new runner type or a new modality is one `register*Runner` call.
 *
 * Lookup falls back from `(service, type)` -> `(service, MOCK)` for the IPC
 * path (logging a warning); the media path requires an exact match.
 */
class RunnerRegistry {
 public:
  using IpcFactory = std::function<std::unique_ptr<runners::IRunner>(
      const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
      ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue)>;

  using MediaFactory = std::function<std::unique_ptr<runners::IRunnerBase>(
      const config::RunnerConfig& config)>;

  RunnerRegistry(const RunnerRegistry&) = delete;
  RunnerRegistry& operator=(const RunnerRegistry&) = delete;

  static RunnerRegistry& instance();

  // ------------------ IPC-loop runners ------------------

  void registerIpcRunner(config::ModelService service,
                         config::ModelRunnerType type, IpcFactory factory);

  /** Returns nullptr if neither an exact match nor a MOCK fallback is
   *  registered. */
  std::unique_ptr<runners::IRunner> createIpc(
      config::ModelService service, config::ModelRunnerType type,
      const config::RunnerConfig& config, ipc::IResultQueue* resultQueue,
      ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue) const;

  bool hasIpc(config::ModelService service,
              config::ModelRunnerType type) const;

  // ------------------ Direct-call media runners ------------------

  void registerMediaRunner(config::ModelService service,
                           config::ModelRunnerType type, MediaFactory factory);

  /** Returns nullptr when no factory is registered; throws when the
   *  registered factory's runner shape doesn't match `Runner`. */
  template <typename Runner>
  std::unique_ptr<Runner> createMedia(
      config::ModelService service, config::ModelRunnerType type,
      const config::RunnerConfig& config) const {
    static_assert(std::is_base_of_v<runners::IRunnerBase, Runner>,
                  "Runner must derive from IRunnerBase");
    auto base = createMediaBase(service, type, config);
    if (!base) return nullptr;
    auto* casted = dynamic_cast<Runner*>(base.get());
    if (!casted) {
      throw std::runtime_error(
          "[RunnerRegistry] media runner shape mismatch for " +
          config::toString(service) + "/" + config::toString(type));
    }
    base.release();
    return std::unique_ptr<Runner>(casted);
  }

  bool hasMedia(config::ModelService service,
                config::ModelRunnerType type) const;

  /** Test-only. */
  void clear();

 private:
  RunnerRegistry() = default;

  std::unique_ptr<runners::IRunnerBase> createMediaBase(
      config::ModelService service, config::ModelRunnerType type,
      const config::RunnerConfig& config) const;

  using Key = std::pair<config::ModelService, config::ModelRunnerType>;

  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      return (static_cast<size_t>(k.first) << 16) ^
             static_cast<size_t>(k.second);
    }
  };

  std::unordered_map<Key, IpcFactory, KeyHash> ipc_factories_;
  std::unordered_map<Key, MediaFactory, KeyHash> media_factories_;
};

}  // namespace tt::utils
