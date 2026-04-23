// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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

namespace tt::ipc {

constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS =
    131072;  // matches Config::MAX_INPUT_TOKENS (128k)
constexpr int DECODE_MAX_TOKEN_IDS = 1;

enum SlotState { EMPTY, FILLED };

struct SlotRingBufferState {
  uint64_t writerIndex;
  uint64_t readerIndex;
};

template <int MaxTokenIds>
struct alignas(8) Message {
  std::atomic<int32_t> state;
  uint32_t maxTokens;
  uint32_t numTokenIds;
  uint32_t taskId;
  uint32_t fastMode;
  uint64_t tokenIds[MaxTokenIds];

  static constexpr size_t K_TOTAL_SIZE = SHM_SLOTS * sizeof(Message);

  bool stateMatches(SlotState state) {
    return this->state.load(std::memory_order_acquire) == state;
  }

  void switchState(SlotState state) {
    this->state.store(state, std::memory_order_release);
  }
};

struct ReadResult {
  uint32_t taskId;
  uint32_t maxTokens;
  bool fastMode = false;
  std::vector<int64_t> tokenIds;
};

template <int MaxTokenIds>
class SlotRingBuffer {
 public:
  using Msg = Message<MaxTokenIds>;

  explicit SlotRingBuffer(const std::string& name)
      : name(name[0] == '/' ? name : "/" + name) {}

  ~SlotRingBuffer() {
    if (memPointer && memPointer != MAP_FAILED) {
      munmap(memPointer, Msg::K_TOTAL_SIZE);
    }
    if (state && state != MAP_FAILED) {
      munmap(state, sizeof(SlotRingBufferState));
    }
  }

  SlotRingBuffer(const SlotRingBuffer&) = delete;
  SlotRingBuffer& operator=(const SlotRingBuffer&) = delete;

  void open() {
    int fd = openOrCreateShm(name, Msg::K_TOTAL_SIZE);
    memPointer = mmap(nullptr, Msg::K_TOTAL_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    int savedErrno = errno;
    ::close(fd);
    if (memPointer == MAP_FAILED) {
      throw std::runtime_error("SlotRingBuffer: mmap failed: " + name + ": " +
                               std::strerror(savedErrno));
    }
    messages = std::span<Msg>(static_cast<Msg*>(memPointer), SHM_SLOTS);
    openState();
    this->writerIndex = state->writerIndex;
    this->readerIndex = state->readerIndex;
  }

  void openState() {
    auto stateName = this->name + "_state";
    int fd = openOrCreateShm(stateName, sizeof(SlotRingBufferState));
    auto memPointerState = mmap(nullptr, sizeof(SlotRingBufferState),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    int savedErrno = errno;
    ::close(fd);
    if (memPointerState == MAP_FAILED) {
      throw std::runtime_error("SlotRingBuffer: mmap failed: " + stateName +
                               ": " + std::strerror(savedErrno));
    }
    // New segments are zero-filled by ftruncate, so no explicit memset needed.
    state = reinterpret_cast<SlotRingBufferState*>(memPointerState);
  }

  void write(uint32_t taskId, const std::vector<int64_t>& tokenIds,
             uint32_t maxTokens, bool fastMode = false) {
    if (static_cast<int>(tokenIds.size()) > MaxTokenIds) {
      throw std::runtime_error("SlotRingBuffer::write: token count " +
                               std::to_string(tokenIds.size()) +
                               " exceeds slot capacity " +
                               std::to_string(MaxTokenIds));
    }

    auto& slot = acquireWriterSlot();

    while (!slot.stateMatches(EMPTY)) {
      std::this_thread::yield();
    }

    slot.taskId = taskId;
    slot.maxTokens = maxTokens;
    slot.fastMode = fastMode ? 1u : 0u;
    slot.numTokenIds = tokenIds.size();
    std::memcpy(slot.tokenIds, tokenIds.data(),
                tokenIds.size() * sizeof(int64_t));

    slot.switchState(FILLED);
    advanceWriterIndex();
  }

  bool tryRead(ReadResult& out) {
    auto& slot = acquireReaderSlot();

    if (!slot.stateMatches(FILLED)) {
      return false;
    }

    out.taskId = slot.taskId;
    out.maxTokens = slot.maxTokens;
    out.fastMode = slot.fastMode != 0u;
    out.tokenIds.assign(slot.tokenIds, slot.tokenIds + slot.numTokenIds);

    slot.switchState(EMPTY);
    advanceReaderIndex();
    return true;
  }
  std::string name;

 private:
  static int openOrCreateShm(const std::string& shmName, size_t size) {
    int fd = shm_open(shmName.c_str(), O_RDWR, 0);
    bool created = false;
    if (fd < 0 && errno == ENOENT) {
      fd = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
      created = (fd >= 0);
    }
    if (fd < 0) {
      int savedErrno = errno;
      throw std::runtime_error(
          "SlotRingBuffer: unable to open shared memory " + shmName + ": " +
          std::strerror(savedErrno));
    }
    // fchmod guards against the creating process's umask stripping bits from
    // the mode passed to shm_open, which would lock out other-UID peers.
    if (created && fchmod(fd, 0666) < 0) {
      int savedErrno = errno;
      ::close(fd);
      throw std::runtime_error("SlotRingBuffer: fchmod failed for " + shmName +
                               ": " + std::strerror(savedErrno));
    }
    if (ftruncate(fd, size) < 0) {
      int savedErrno = errno;
      ::close(fd);
      throw std::runtime_error("SlotRingBuffer: ftruncate failed for " +
                               shmName + ": " +
                               std::strerror(savedErrno));
    }
    return fd;
  }

  Msg& acquireReaderSlot() { return messages[readerIndex]; }
  Msg& acquireWriterSlot() { return messages[writerIndex]; }

  void advanceReaderIndex() {
    readerIndex = (readerIndex + 1) % SHM_SLOTS;
    state->readerIndex = readerIndex;
  }

  void advanceWriterIndex() {
    writerIndex = (writerIndex + 1) % SHM_SLOTS;
    state->writerIndex = writerIndex;
  }

  void* memPointer = nullptr;
  uint64_t writerIndex = 0;
  uint64_t readerIndex = 0;
  std::span<Msg> messages;
  SlotRingBufferState* state = nullptr;
};

using PrefillSlotBuffer = SlotRingBuffer<PREFILL_MAX_TOKEN_IDS>;
using DecodeSlotBuffer = SlotRingBuffer<DECODE_MAX_TOKEN_IDS>;

}  // namespace tt::ipc
