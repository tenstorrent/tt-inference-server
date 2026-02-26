#pragma once

#include "profiling/tracy.hpp"
#include <optional>
#include <mutex>
#include <unordered_map>

template <typename Key, typename Value>
class ConcurrentMap {
public:
    ConcurrentMap() = default;
    ~ConcurrentMap() = default;

    void insert(const Key& key, const Value& value) {
        std::lock_guard lock(mutex_);
        map_[key] = value;
    }

    std::optional<Value> get(const Key& key) {
        std::lock_guard lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void erase(const Key& key) {
        std::lock_guard lock(mutex_);
        map_.erase(key);
    }

    bool contains(const Key& key) {
        std::lock_guard lock(mutex_);
        return map_.find(key) != map_.end();
    }

    ConcurrentMap(const ConcurrentMap&) = delete;
    ConcurrentMap& operator=(const ConcurrentMap&) = delete;

private:
    std::unordered_map<Key, Value> map_;
    TracyLockable(std::mutex, mutex_);
};