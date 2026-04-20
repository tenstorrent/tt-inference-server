// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#pragma once

#include <mutex>
#include <optional>
#include <unordered_map>

#include "profiling/tracy.hpp"

namespace tt::utils {

template <typename Key, typename Value>
class ConcurrentMap {
 public:
  ConcurrentMap() = default;
  ~ConcurrentMap() = default;

  void insert(const Key& key, const Value& value) {
    std::lock_guard lock(mutex);
    map_[key] = value;
  }

  void insert(const Key& key, Value&& value) {
    std::lock_guard lock(mutex);
    map_[key] = std::move(value);
  }

  std::optional<Value> get(const Key& key) const {
    std::lock_guard lock(mutex);
    auto it = map_.find(key);
    if (it != map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void erase(const Key& key) {
    std::lock_guard lock(mutex);
    map_.erase(key);
  }

  std::optional<Value> take(const Key& key) {
    std::lock_guard lock(mutex);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return std::nullopt;
    }
    auto value = std::move(it->second);
    map_.erase(it);
    return value;
  }

  template <typename Pred>
  std::optional<Value> takeIf(const Key& key, Pred&& pred) {
    std::lock_guard lock(mutex);
    auto it = map_.find(key);
    if (it == map_.end()) {
      return std::nullopt;
    }
    if (!pred(it->second)) {
      return std::nullopt;
    }
    auto value = std::move(it->second);
    map_.erase(it);
    return value;
  }

  bool contains(const Key& key) const {
    std::lock_guard lock(mutex);
    return map_.find(key) != map_.end();
  }

  void clear() {
    std::lock_guard lock(mutex);
    map_.clear();
  }

  size_t size() const {
    std::lock_guard lock(mutex);
    return map_.size();
  }

  template <typename Func>
  bool modify(const Key& key, Func&& func) {
    std::lock_guard lock(mutex);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    func(it->second);
    return true;
  }

  template <typename Func>
  void forEach(Func&& func) {
    std::lock_guard lock(mutex);
    for (auto& [key, value] : map_) {
      func(key, value);
    }
  }

  ConcurrentMap(const ConcurrentMap&) = delete;
  ConcurrentMap& operator=(const ConcurrentMap&) = delete;

 private:
  std::unordered_map<Key, Value> map_;
  mutable TRACY_LOCKABLE(std::mutex, mutex);
};

}  // namespace tt::utils
