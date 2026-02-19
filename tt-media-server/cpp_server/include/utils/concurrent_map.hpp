#pragma once

#include <optional>
#include <mutex>
#include <unordered_map>

using namespace std;

template <typename Key, typename Value>
class ConcurrentMap {
public:
    ConcurrentMap() = default;
    ~ConcurrentMap() = default;

    void insert(const Key& key, const Value& value) {
        lock_guard<mutex> lock(mutex_);
        map_[key] = value;
    }

    optional<Value> get(const Key& key) {
        lock_guard<mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            return it->second;
        }
        return nullopt;
    }

    void erase(const Key& key) {
        lock_guard<mutex> lock(mutex_);
        map_.erase(key);
    }

    bool contains(const Key& key) {
        lock_guard<mutex> lock(mutex_);
        return map_.find(key) != map_.end();
    }

    ConcurrentMap(const ConcurrentMap&) = delete;
    ConcurrentMap& operator=(const ConcurrentMap&) = delete;

private:
    unordered_map<Key, Value> map_;
    mutable mutex mutex_;
};