// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

#include <mutex>
#include <optional>
#include <unordered_map>

#include "profiling/tracy.hpp"  // NOLINT(misc-include-cleaner) - TracyLockable macro

template <typename Key, typename Value>
class ConcurrentMap {
 public:
  ConcurrentMap() = default;
  ~ConcurrentMap() = default;

  void insert(const Key& key, const Value& value) {
    std::lock_guard lock(mutex);
    map[key] = value;
  }

  std::optional<Value> get(const Key& key) {
    std::lock_guard lock(mutex);
    auto it = map.find(key);
    if (it != map.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void erase(const Key& key) {
    std::lock_guard lock(mutex);
    map.erase(key);
  }

  std::optional<Value> take(const Key& key) {
    std::lock_guard lock(mutex);
    auto it = map.find(key);
    if (it == map.end()) {
      return std::nullopt;
    }
    auto value = std::move(it->second);
    map.erase(it);
    return value;
  }

  bool contains(const Key& key) {
    std::lock_guard lock(mutex);
    return map.find(key) != map.end();
  }

  void clear() {
    std::lock_guard lock(mutex);
    map.clear();
  }

  template <typename Func>
  void forEach(Func&& func) {
    std::lock_guard lock(mutex);
    for (auto& [key, value] : map) {
      func(key, value);
    }
  }

  ConcurrentMap(const ConcurrentMap&) = delete;
  ConcurrentMap& operator=(const ConcurrentMap&) = delete;

 private:
  std::unordered_map<Key, Value> map;
  TracyLockable(std::mutex, mutex);  // NOLINT(readability-identifier-naming)
};
