// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

namespace tt::ipc {

/**
 * Lock-free Single-Producer Single-Consumer Ring Buffer in Shared Memory.
 * Used for high-performance IPC between worker processes and main process.
 *
 * Performance: ~1-5 µs per token (vs ~0.1-1 µs for direct callback)
 * But allows process isolation for different environment variables per worker.
 */

// Fixed-size token structure for zero-copy transfer
struct alignas(64) SharedToken {
    uint32_t token_index;       // Token index in sequence
    uint32_t flags;             // Bit flags: is_final, is_error, etc.
    uint64_t token_id;          // Token ID
    char task_id[56];       // Task ID
    char padding[8];            // Padding to reach 128 bytes

    static constexpr uint32_t FLAG_FINAL = 1;
    static constexpr uint32_t FLAG_ERROR = 2;
    static constexpr uint32_t FLAG_DONE = 4;

    bool is_final() const { return flags & FLAG_FINAL; }
    bool is_error() const { return flags & FLAG_ERROR; }
    bool is_done() const { return flags & FLAG_DONE; }
};

static_assert(sizeof(SharedToken) == 128, "SharedToken must be 128 bytes for cache alignment");

/**
 * Shared memory ring buffer header.
 * Lives at the start of the shared memory region.
 */
struct alignas(64) RingBufferHeader {
    // Cache line 1: Write position (producer)
    alignas(64) std::atomic<uint64_t> write_pos{0};

    // Cache line 2: Read position (consumer)
    alignas(64) std::atomic<uint64_t> read_pos{0};

    // Cache line 3: Metadata
    alignas(64) uint64_t capacity{0};
    uint64_t token_offset{0};  // Offset to token array
    uint32_t version{1};
    uint32_t worker_count{0};
    std::atomic<bool> shutdown{false};
    char padding[32];
};

static_assert(sizeof(RingBufferHeader) == 192, "Header size check");

/**
 * SPSC Ring Buffer for token streaming.
 * Lock-free for maximum performance.
 */
template<size_t CAPACITY = 65536>
class TokenRingBuffer {
public:
    static constexpr size_t BUFFER_CAPACITY = CAPACITY;

    // Calculate total shared memory size needed
    static constexpr size_t shared_memory_size() {
        return sizeof(RingBufferHeader) + sizeof(SharedToken) * CAPACITY;
    }

    /**
     * Create a new ring buffer in shared memory.
     * @param name Shared memory name (e.g., "/tt_tokens_0")
     * @param create If true, create new; if false, attach to existing
     */
    TokenRingBuffer(const std::string& name, bool create)
        : shm_name_(name), is_owner_(create) {

        int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
        mode_t mode = S_IRUSR | S_IWUSR;

        shm_fd_ = shm_open(name.c_str(), flags, mode);
        if (shm_fd_ < 0) {
            throw std::runtime_error("Failed to open shared memory: " + name);
        }

        size_t size = shared_memory_size();

        if (create) {
            if (ftruncate(shm_fd_, size) < 0) {
                close(shm_fd_);
                shm_unlink(name.c_str());
                throw std::runtime_error("Failed to resize shared memory");
            }
        }

        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (ptr == MAP_FAILED) {
            close(shm_fd_);
            if (create) shm_unlink(name.c_str());
            throw std::runtime_error("Failed to mmap shared memory");
        }

        shm_ptr_ = ptr;
        header_ = reinterpret_cast<RingBufferHeader*>(ptr);
        tokens_ = reinterpret_cast<SharedToken*>(
            reinterpret_cast<char*>(ptr) + sizeof(RingBufferHeader)
        );

        if (create) {
            // Initialize header
            new (header_) RingBufferHeader();
            header_->capacity = CAPACITY;
            header_->token_offset = sizeof(RingBufferHeader);

            // Zero-initialize tokens
            std::memset(tokens_, 0, sizeof(SharedToken) * CAPACITY);
        }
    }

    ~TokenRingBuffer() {
        if (shm_ptr_) {
            munmap(shm_ptr_, shared_memory_size());
        }
        if (shm_fd_ >= 0) {
            close(shm_fd_);
        }
        if (is_owner_) {
            shm_unlink(shm_name_.c_str());
        }
    }

    // Prevent copying
    TokenRingBuffer(const TokenRingBuffer&) = delete;
    TokenRingBuffer& operator=(const TokenRingBuffer&) = delete;

    // Move is OK
    TokenRingBuffer(TokenRingBuffer&& other) noexcept
        : shm_name_(std::move(other.shm_name_))
        , shm_fd_(other.shm_fd_)
        , shm_ptr_(other.shm_ptr_)
        , header_(other.header_)
        , tokens_(other.tokens_)
        , is_owner_(other.is_owner_) {
        other.shm_fd_ = -1;
        other.shm_ptr_ = nullptr;
        other.header_ = nullptr;
        other.tokens_ = nullptr;
        other.is_owner_ = false;
    }

    /**
     * Push a token (producer side).
     * Returns false if buffer is full.
     */
    bool push(const SharedToken& token) {
        uint64_t write = header_->write_pos.load(std::memory_order_relaxed);
        uint64_t read = header_->read_pos.load(std::memory_order_acquire);

        // Check if full
        if (write - read >= CAPACITY) {
            return false;
        }

        // Write token
        size_t idx = write % CAPACITY;
        tokens_[idx] = token;

        // Publish
        header_->write_pos.store(write + 1, std::memory_order_release);
        return true;
    }

    /**
     * Pop a token (consumer side).
     * Returns false if buffer is empty.
     */
    bool pop(SharedToken& token) {
        uint64_t read = header_->read_pos.load(std::memory_order_relaxed);
        uint64_t write = header_->write_pos.load(std::memory_order_acquire);

        // Check if empty
        if (read >= write) {
            return false;
        }

        // Read token
        size_t idx = read % CAPACITY;
        token = tokens_[idx];

        // Advance read position
        header_->read_pos.store(read + 1, std::memory_order_release);
        return true;
    }

    /**
     * Peek at next token without consuming.
     */
    bool peek(SharedToken& token) const {
        uint64_t read = header_->read_pos.load(std::memory_order_relaxed);
        uint64_t write = header_->write_pos.load(std::memory_order_acquire);

        if (read >= write) {
            return false;
        }

        token = tokens_[read % CAPACITY];
        return true;
    }

    /**
     * Get number of tokens available to read.
     */
    size_t available() const {
        uint64_t write = header_->write_pos.load(std::memory_order_acquire);
        uint64_t read = header_->read_pos.load(std::memory_order_relaxed);
        return static_cast<size_t>(write - read);
    }

    /**
     * Check if empty.
     */
    bool empty() const {
        return available() == 0;
    }

    /**
     * Signal shutdown.
     */
    void shutdown() {
        header_->shutdown.store(true, std::memory_order_release);
    }

    /**
     * Check if shutdown requested.
     */
    bool is_shutdown() const {
        return header_->shutdown.load(std::memory_order_acquire);
    }

    /**
     * Get the shared memory name.
     */
    const std::string& name() const { return shm_name_; }

private:
    std::string shm_name_;
    int shm_fd_ = -1;
    void* shm_ptr_ = nullptr;
    RingBufferHeader* header_ = nullptr;
    SharedToken* tokens_ = nullptr;
    bool is_owner_ = false;
};

// Default ring buffer type
using DefaultTokenRingBuffer = TokenRingBuffer<65536>;

} // namespace tt::ipc
