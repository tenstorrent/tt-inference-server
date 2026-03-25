// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sp_pipeline {

constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS =
    131072;  // matches Config::MAX_INPUT_TOKENS (128k)
constexpr int DECODE_MAX_TOKEN_IDS = 1;

enum SlotState { FREE, TAKEN };

struct SharedMemoryState {
  uint64_t cursor;
};

template <int MaxTokenIds>
struct Message {
  std::atomic<int32_t> state;
  uint32_t maxTokens;
  uint32_t numTokenIds;
  char taskId[36];
  uint64_t tokenIds[MaxTokenIds];

  bool stateMatches(SlotState s) const {
    return state.load(std::memory_order_acquire) == static_cast<int32_t>(s);
  }

  void switchState(SlotState s) {
    state.store(static_cast<int32_t>(s), std::memory_order_release);
  }
};

static_assert(sizeof(Message<PREFILL_MAX_TOKEN_IDS>) ==
              48 + PREFILL_MAX_TOKEN_IDS * sizeof(uint64_t));
static_assert(sizeof(Message<DECODE_MAX_TOKEN_IDS>) ==
              48 + DECODE_MAX_TOKEN_IDS * sizeof(uint64_t));

struct ReadResult {
  std::string taskId;
  uint32_t maxTokens;
  std::vector<int64_t> tokenIds;
};

template <typename SlotType>
class SharedMemory {
 public:
  static constexpr size_t k_total_size = SHM_SLOTS * sizeof(SlotType);

  explicit SharedMemory(const std::string& name, bool create = false,
                        bool owner = true)
      : name(name[0] == '/' ? name : "/" + name),
        create_(create),
        owner_(owner) {}

  ~SharedMemory() {
    if (memPointer && memPointer != MAP_FAILED) {
      munmap(memPointer, k_total_size);
    }
    if (owner_) {
      shm_unlink(name.c_str());
    }
    if (state && state != MAP_FAILED) {
      munmap(state, sizeof(SharedMemoryState));
    }
  }

  SharedMemory(const SharedMemory&) = delete;
  SharedMemory& operator=(const SharedMemory&) = delete;

  void open() {
    if (create_) {
      shm_unlink(name.c_str());
    }
    int flags = create_ ? (O_RDWR | O_CREAT) : O_RDWR;
    int fd = shm_open(name.c_str(), flags, 0666);
    if (fd < 0) {
      throw std::runtime_error("SharedMemory: unable to open shared memory: " +
                               name);
    }
    if (create_) {
      if (ftruncate(fd, static_cast<off_t>(k_total_size)) < 0) {
        ::close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error("SharedMemory: ftruncate failed: " + name);
      }
    }
    memPointer =
        mmap(nullptr, k_total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (memPointer == MAP_FAILED) {
      ::close(fd);
      if (create_) shm_unlink(name.c_str());
      throw std::runtime_error("SharedMemory: mmap failed: " + name);
    }
    ::close(fd);

    messages =
        std::span<SlotType>(static_cast<SlotType*>(memPointer), SHM_SLOTS);
    openState();
    this->current = state->cursor;
  }

  void openState() {
    auto name = this->name + "_state";
    int fd;
    fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
      fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
      ftruncate(fd, sizeof(SharedMemoryState));
      auto memPointerState = mmap(nullptr, sizeof(SharedMemoryState),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      memset(memPointerState, 0, sizeof(SharedMemoryState));
      ::close(fd);
      state = reinterpret_cast<SharedMemoryState*>(memPointerState);
      return;
    }
    auto memPointerState = mmap(nullptr, sizeof(SharedMemoryState),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    state = reinterpret_cast<SharedMemoryState*>(memPointerState);
    return;
  }

  void write(const std::string& uuid, const std::vector<int64_t>& tokenIds,
             uint32_t maxTokens) {
    if (tokenIds.size() * sizeof(int64_t) > sizeof(SlotType::tokenIds)) {
      throw std::runtime_error("SharedMemory::write: token count " +
                               std::to_string(tokenIds.size()) +
                               " exceeds slot capacity");
    }
    auto& msg = acquireSlot();
    while (!msg.stateMatches(FREE)) {
      std::this_thread::yield();
    }
    msg.maxTokens = maxTokens;
    msg.numTokenIds = tokenIds.size();
    std::memcpy(msg.taskId, uuid.data(), 36);
    std::memcpy(msg.tokenIds, tokenIds.data(),
                tokenIds.size() * sizeof(int64_t));
    msg.switchState(TAKEN);
    advanceCurrent();
  }

  bool tryRead(ReadResult& out) {
    auto& msg = acquireSlot();
    if (!msg.stateMatches(TAKEN)) return false;
    out.taskId.assign(msg.taskId, 36);
    out.maxTokens = msg.maxTokens;
    out.tokenIds.assign(msg.tokenIds, msg.tokenIds + msg.numTokenIds);
    msg.switchState(FREE);
    advanceCurrent();
    return true;
  }

  std::string name;

 private:
  SlotType& acquireSlot() { return messages[current]; }

  void advanceCurrent() {
    current = (current + 1) % SHM_SLOTS;
    updateState();
  }

  void updateState() { state->cursor = current; }

  bool create_;
  bool owner_;
  void* memPointer = nullptr;
  uint64_t current = 0;
  std::span<SlotType> messages;
  SharedMemoryState* state = nullptr;
};

using PrefillSharedMemory = SharedMemory<Message<PREFILL_MAX_TOKEN_IDS>>;
using DecodeSharedMemory = SharedMemory<Message<DECODE_MAX_TOKEN_IDS>>;

}  // namespace sp_pipeline
