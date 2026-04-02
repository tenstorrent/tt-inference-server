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

namespace tt::ipc {

constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS =
    131072;  // matches Config::MAX_INPUT_TOKENS (128k)
constexpr int DECODE_MAX_TOKEN_IDS = 1;

enum SlotState { FREE, TAKEN };

struct SlotRingBufferState {
  uint64_t cursor;
};

template <int MaxTokenIds>
struct Message {
  std::atomic<int32_t> state;
  uint32_t maxTokens;
  uint32_t numTokenIds;
  uint32_t fastMode;
  char taskId[36];
  uint32_t _reserved;
  uint64_t tokenIds[MaxTokenIds];

  static constexpr int K_MESSAGE_SIZE = 56 + MaxTokenIds * sizeof(uint64_t);
  static constexpr int K_TOTAL_SIZE = SHM_SLOTS * K_MESSAGE_SIZE;

  bool stateMatches(SlotState state) {
    return this->state.load(std::memory_order_acquire) == state;
  }

  void switchState(SlotState state) {
    this->state.store(state, std::memory_order_release);
  }
};

static_assert(sizeof(Message<PREFILL_MAX_TOKEN_IDS>) ==
              Message<PREFILL_MAX_TOKEN_IDS>::K_MESSAGE_SIZE);
static_assert(sizeof(Message<DECODE_MAX_TOKEN_IDS>) ==
              Message<DECODE_MAX_TOKEN_IDS>::K_MESSAGE_SIZE);

struct ReadResult {
  std::string taskId;
  uint32_t maxTokens;
  std::vector<int64_t> tokenIds;
  bool fastMode = false;
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
    int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
      throw std::runtime_error(
          "SlotRingBuffer: unable to open shared memory: " + name);
    }

    memPointer = mmap(nullptr, Msg::K_TOTAL_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (memPointer == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("SlotRingBuffer: mmap failed: " + name);
    }
    ::close(fd);

    messages = std::span<Msg>(static_cast<Msg*>(memPointer), SHM_SLOTS);
    openState();
    this->current = state->cursor;
  }

  void openState() {
    auto name = this->name + "_state";
    int fd;
    fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
      fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
      ftruncate(fd, sizeof(SlotRingBufferState));
      auto memPointerState = mmap(nullptr, sizeof(SlotRingBufferState),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      memset(memPointerState, 0, sizeof(SlotRingBufferState));
      ::close(fd);
      state = reinterpret_cast<SlotRingBufferState*>(memPointerState);
      return;
    }
    auto memPointerState = mmap(nullptr, sizeof(SlotRingBufferState),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    state = reinterpret_cast<SlotRingBufferState*>(memPointerState);
    return;
  }

  void write(const std::string& uuid, const std::vector<int64_t>& tokenIds,
             uint32_t maxTokens, bool fastMode = false) {
    if (static_cast<int>(tokenIds.size()) > MaxTokenIds) {
      throw std::runtime_error("SlotRingBuffer::write: token count " +
                               std::to_string(tokenIds.size()) +
                               " exceeds slot capacity " +
                               std::to_string(MaxTokenIds));
    }

    auto& msg = acquireMsg();

    while (!msg.stateMatches(FREE)) {
      std::this_thread::yield();
    }

    msg.maxTokens = maxTokens;
    msg.numTokenIds = tokenIds.size();
    msg.fastMode = fastMode ? 1 : 0;
    std::memcpy(msg.taskId, uuid.data(), 36);
    std::memcpy(msg.tokenIds, tokenIds.data(),
                tokenIds.size() * sizeof(int64_t));

    msg.switchState(TAKEN);
    advanceCurrent();
  }

  bool tryRead(ReadResult& out) {
    auto& msg = acquireMsg();

    if (!msg.stateMatches(TAKEN)) {
      return false;
    }

    out.taskId.assign(msg.taskId, 36);
    out.maxTokens = msg.maxTokens;
    out.tokenIds.assign(msg.tokenIds, msg.tokenIds + msg.numTokenIds);
    out.fastMode = msg.fastMode != 0;

    msg.switchState(FREE);
    advanceCurrent();
    return true;
  }
  std::string name;

 private:
  Msg& acquireMsg() { return messages[current]; }

  void advanceCurrent() {
    current = (current + 1) % SHM_SLOTS;
    updateState();
  }

  void updateState() { state->cursor = current; }

  void* memPointer = nullptr;
  uint64_t current = 0;
  std::span<Msg> messages;
  SlotRingBufferState* state = nullptr;
};

using PrefillSlotBuffer = SlotRingBuffer<PREFILL_MAX_TOKEN_IDS>;
using DecodeSlotBuffer = SlotRingBuffer<DECODE_MAX_TOKEN_IDS>;

}  // namespace tt::ipc
