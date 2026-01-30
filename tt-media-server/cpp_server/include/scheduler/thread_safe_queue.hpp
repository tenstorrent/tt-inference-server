// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>
#include <chrono>

namespace tt::scheduler {

/**
 * Thread-safe queue for inter-thread communication.
 * Similar to Python's multiprocessing.Queue but for C++ threads.
 */
template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 10000)
        : max_size_(max_size), shutdown_(false) {}

    /**
     * Push an item to the queue. Blocks if queue is full.
     * @return true if item was pushed, false if queue is shutdown
     */
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] {
            return queue_.size() < max_size_ || shutdown_;
        });

        if (shutdown_) return false;

        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /**
     * Try to push an item without blocking.
     * @return true if successful, false if queue is full or shutdown
     */
    bool try_push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_ || queue_.size() >= max_size_) return false;
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /**
     * Pop an item from the queue. Blocks if queue is empty.
     * @return the item, or nullopt if queue is shutdown
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] {
            return !queue_.empty() || shutdown_;
        });

        if (shutdown_ && queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    /**
     * Try to pop an item without blocking.
     * @return the item, or nullopt if queue is empty
     */
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    /**
     * Pop with timeout.
     * @return the item, or nullopt if timeout or shutdown
     */
    template<typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, timeout, [this] {
            return !queue_.empty() || shutdown_;
        })) {
            return std::nullopt;
        }

        if (shutdown_ && queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    /**
     * Get current queue size.
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * Check if queue is empty.
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * Shutdown the queue, unblocking all waiting threads.
     */
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /**
     * Check if queue is shutdown.
     */
    bool is_shutdown() const {
        return shutdown_.load();
    }

    /**
     * Clear the queue.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
        not_full_.notify_all();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t max_size_;
    std::atomic<bool> shutdown_;
};

} // namespace tt::scheduler
