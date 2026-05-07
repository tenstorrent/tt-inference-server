// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "config/types.hpp"

namespace tt::utils {

/**
 * Registry for in-process media runners (image today, audio/video next).
 *
 * Each modality instantiates its own typed registry via
 * `MediaRunnerRegistry<Runner, Config>::instance()` and registers concrete
 * runners keyed by `ModelRunnerType`. Lookup is exact: returns nullptr when
 * no factory matches.
 *
 * Thread-safe.
 */
template <typename Runner, typename Config>
class MediaRunnerRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Runner>(const Config&)>;

  MediaRunnerRegistry(const MediaRunnerRegistry&) = delete;
  MediaRunnerRegistry& operator=(const MediaRunnerRegistry&) = delete;

  static MediaRunnerRegistry& instance() {
    static MediaRunnerRegistry registry;
    return registry;
  }

  void registerRunner(config::ModelRunnerType type, Factory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[type] = std::move(factory);
  }

  std::unique_ptr<Runner> create(config::ModelRunnerType type,
                                 const Config& cfg) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = factories_.find(type);
    if (it == factories_.end()) return nullptr;
    return it->second(cfg);
  }

  bool has(config::ModelRunnerType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return factories_.find(type) != factories_.end();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_.clear();
  }

 private:
  MediaRunnerRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<config::ModelRunnerType, Factory> factories_;
};

}  // namespace tt::utils
