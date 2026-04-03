// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

#include <mutex>
#include <optional>
#include <unordered_map>

#include "profiling/tracy.hpp"

template <typename Key, typename Value>
class ConcurrentMap {
 public:
  ConcurrentMap() = default;
  ~ConcurrentMap() = default;

  void insert(const Key& key, const Value& value) {
    std::lock_guard lock(mutex);
    map_[key] = value;
  }

  std::optional<Value> get(const Key& key) {
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

  bool contains(const Key& key) {
    std::lock_guard lock(mutex);
    return map_.find(key) != map_.end();
  }

  void clear() {
    std::lock_guard lock(mutex);
    map_.clear();
  }

  size_t size() {
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
  TRACY_LOCKABLE(std::mutex, mutex);
};
