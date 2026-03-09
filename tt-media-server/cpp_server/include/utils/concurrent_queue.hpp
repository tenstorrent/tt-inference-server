// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#pragma once

#include "profiling/tracy.hpp"
#include <mutex>
#include <vector>

template <typename T>
class ConcurrentQueue {
public:
    ConcurrentQueue() = default;
    ~ConcurrentQueue() = default;

    void push(const T& value) {
        std::lock_guard lock(mutex_);
        pending_.push_back(value);
    }

    std::vector<T> drain() {
        std::lock_guard lock(mutex_);
        std::vector<T> out;
        out.swap(pending_);
        return out;
    }

    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

private:
    std::vector<T> pending_;
    TracyLockable(std::mutex, mutex_);
};
