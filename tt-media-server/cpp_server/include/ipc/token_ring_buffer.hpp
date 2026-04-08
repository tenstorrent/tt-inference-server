// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

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
  uint32_t token_index;  // Token index in sequence
  uint32_t flags;        // Bit flags: is_final, is_error, etc.
  uint64_t token_id;     // Token ID
  uint32_t task_id;      // Task ID
  char padding[44];      // Padding to reach 64 bytes

  static constexpr uint32_t FLAG_FINAL = 1;
  static constexpr uint32_t FLAG_ERROR = 2;
  static constexpr uint32_t FLAG_DONE = 4;

  bool isFinal() const { return flags & FLAG_FINAL; }
  bool isError() const { return flags & FLAG_ERROR; }
  bool isDone() const { return flags & FLAG_DONE; }
};

static_assert(sizeof(SharedToken) == 64,
              "SharedToken must be 64 bytes for cache alignment");

struct SharedEmbedding {};

namespace detail {

inline int futexWait(std::atomic<uint32_t>& futexWord, uint32_t expected,
                     const struct timespec* timeout = nullptr) {
  return syscall(SYS_futex, reinterpret_cast<uint32_t*>(&futexWord), FUTEX_WAIT,
                 expected, timeout, nullptr, 0);
}

inline void futexWake(std::atomic<uint32_t>& futexWord, int count = 1) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(&futexWord), FUTEX_WAKE, count,
          nullptr, nullptr, 0);
}

}  // namespace detail

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
  uint64_t token_offset{0};
  uint32_t version{1};
  uint32_t worker_count{0};
  std::atomic<bool> shutdown{false};
  std::atomic<uint32_t> push_notify{0};  // Futex word for blocking pop
  char padding[28];
};

static_assert(sizeof(RingBufferHeader) == 192, "Header size check");

/**
 * SPSC Ring Buffer for token streaming.
 * Lock-free for maximum performance.
 */
template <size_t CAPACITY = 65536>
class TokenRingBuffer {
 public:
  static constexpr size_t BUFFER_CAPACITY = CAPACITY;

  // Calculate total shared memory size needed
  static constexpr size_t sharedMemorySize() {
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

    if (create) {
      // Unlink existing shared memory to ensure clean state
      shm_unlink(name.c_str());
    }

    shm_fd_ = shm_open(name.c_str(), flags, mode);
    if (shm_fd_ < 0) {
      throw std::runtime_error("Failed to open shared memory: " + name);
    }

    size_t size = sharedMemorySize();

    if (create) {
      if (ftruncate(shm_fd_, size) < 0) {
        close(shm_fd_);
        shm_unlink(name.c_str());
        throw std::runtime_error("Failed to resize shared memory");
      }
    }

    void* ptr =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (ptr == MAP_FAILED) {
      close(shm_fd_);
      if (create) shm_unlink(name.c_str());
      throw std::runtime_error("Failed to mmap shared memory");
    }

    shm_ptr_ = ptr;
    header_ = reinterpret_cast<RingBufferHeader*>(ptr);
    tokens_ = reinterpret_cast<SharedToken*>(reinterpret_cast<char*>(ptr) +
                                             sizeof(RingBufferHeader));

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
      munmap(shm_ptr_, sharedMemorySize());
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
      : shm_name_(std::move(other.shm_name_)),
        shm_fd_(other.shm_fd_),
        shm_ptr_(other.shm_ptr_),
        header_(other.header_),
        tokens_(other.tokens_),
        is_owner_(other.is_owner_) {
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

    if (write - read >= CAPACITY) {
      return false;
    }

    size_t idx = write % CAPACITY;
    tokens_[idx] = token;

    header_->write_pos.store(write + 1, std::memory_order_release);

    header_->push_notify.fetch_add(1, std::memory_order_release);
    detail::futexWake(header_->push_notify);
    return true;
  }

  /**
   * Pop a token (consumer side). Non-blocking.
   * Returns false if buffer is empty.
   */
  bool pop(SharedToken& token) {
    uint64_t read = header_->read_pos.load(std::memory_order_relaxed);
    uint64_t write = header_->write_pos.load(std::memory_order_acquire);

    if (read >= write) {
      return false;
    }

    size_t idx = read % CAPACITY;
    token = tokens_[idx];

    header_->read_pos.store(read + 1, std::memory_order_release);
    return true;
  }

  /**
   * Pop a token (consumer side). Blocks until a token is available
   * or shutdown is signaled. Returns false only on shutdown with
   * an empty buffer.
   */
  bool blockingPop(SharedToken& token) {
    while (!isShutdown()) {
      if (pop(token)) {
        return true;
      }

      uint32_t snapshot = header_->push_notify.load(std::memory_order_acquire);

      // Re-check after capturing snapshot to avoid missed-wake race
      if (pop(token)) {
        return true;
      }

      struct timespec timeout = {
          0, 1'000'000};  // 1 ms — periodic wake for shutdown check
      detail::futexWait(header_->push_notify, snapshot, &timeout);
    }
    return pop(token);
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
  bool empty() const { return available() == 0; }

  /**
   * Signal shutdown and wake any blocked consumer.
   */
  void shutdown() {
    header_->shutdown.store(true, std::memory_order_release);
    header_->push_notify.fetch_add(1, std::memory_order_release);
    detail::futexWake(header_->push_notify, INT_MAX);
  }

  /**
   * Check if shutdown requested.
   */
  bool isShutdown() const {
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

}  // namespace tt::ipc
