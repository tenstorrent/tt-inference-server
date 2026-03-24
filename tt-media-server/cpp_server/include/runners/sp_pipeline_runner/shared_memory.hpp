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

#include "domain/manage_memory.hpp"

namespace sp_pipeline {

constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS =
    131072;  // matches Config::MAX_INPUT_TOKENS (128k)
constexpr int DECODE_MAX_TOKEN_IDS = 1;

enum SlotState { FREE, TAKEN };

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

constexpr uint32_t MEMORY_RESULT_MAX_KV_DESTINATIONS = 128;

struct MemoryRequestSlot {
  std::atomic<int32_t> state;
  int32_t input_seq_len{};
  uint8_t action{};
  uint8_t memory_layout{};
  uint8_t reserved[2]{};
  char taskId[36]{};

  bool stateMatches(SlotState s) const {
    return state.load(std::memory_order_acquire) == static_cast<int32_t>(s);
  }

  void switchState(SlotState s) {
    state.store(static_cast<int32_t>(s), std::memory_order_release);
  }
};

static_assert(sizeof(MemoryRequestSlot) == 48);

struct MemoryResultSlot {
  std::atomic<int32_t> state;
  uint32_t success{};
  uint32_t num_destinations{};
  char taskId[36]{};
  tt::domain::KvDestination destinations[MEMORY_RESULT_MAX_KV_DESTINATIONS]{};

  bool stateMatches(SlotState s) const {
    return state.load(std::memory_order_acquire) == static_cast<int32_t>(s);
  }

  void switchState(SlotState s) {
    state.store(static_cast<int32_t>(s), std::memory_order_release);
  }
};

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

    if (create_) {
      std::memset(memPointer, 0, k_total_size);
    }
    messages =
        std::span<SlotType>(static_cast<SlotType*>(memPointer), SHM_SLOTS);
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

  void writeRequest(const tt::domain::ManageMemoryTask& task) {
    auto& msg = acquireSlot();
    while (!msg.stateMatches(FREE)) {
      std::this_thread::yield();
    }
    msg.input_seq_len = task.input_seq_len;
    msg.action = static_cast<uint8_t>(task.action);
    msg.memory_layout = static_cast<uint8_t>(task.memory_layout);
    const size_t n = std::min(task.task_id.id.size(), static_cast<size_t>(36));
    std::memcpy(msg.taskId, task.task_id.id.data(), n);
    if (n < 36) std::memset(msg.taskId + n, 0, 36 - n);
    msg.switchState(TAKEN);
    advanceCurrent();
  }

  bool tryReadRequest(tt::domain::ManageMemoryTask& out) {
    auto& msg = acquireSlot();
    if (!msg.stateMatches(TAKEN)) return false;
    out.task_id =
        tt::domain::TaskID::ipcDeserialize(msg.taskId, sizeof(msg.taskId));
    out.input_seq_len = msg.input_seq_len;
    out.action = static_cast<tt::domain::MemoryManagementAction>(msg.action);
    out.memory_layout =
        static_cast<tt::domain::KvMemoryLayout>(msg.memory_layout);
    msg.switchState(FREE);
    advanceCurrent();
    return true;
  }

  void writeResult(const tt::domain::ManageMemoryResult& result) {
    auto& msg = acquireSlot();
    while (!msg.stateMatches(FREE)) {
      std::this_thread::yield();
    }
    msg.success =
        (result.status == tt::domain::ManageMemoryStatus::SUCCESS) ? 1u : 0u;
    const size_t n =
        std::min(result.memory_locations.size(),
                 static_cast<size_t>(MEMORY_RESULT_MAX_KV_DESTINATIONS));
    msg.num_destinations = static_cast<uint32_t>(n);
    const size_t tid_n =
        std::min(result.task_id.id.size(), static_cast<size_t>(36));
    std::memcpy(msg.taskId, result.task_id.id.data(), tid_n);
    if (tid_n < 36) std::memset(msg.taskId + tid_n, 0, 36 - tid_n);
    for (uint32_t i = 0; i < msg.num_destinations; ++i) {
      msg.destinations[i].dram_address =
          result.memory_locations[i].dram_address;
      msg.destinations[i].semaphore_address =
          result.memory_locations[i].semaphore_address;
    }
    msg.switchState(TAKEN);
    advanceCurrent();
  }

  bool tryReadResult(tt::domain::ManageMemoryResult& out) {
    auto& msg = acquireSlot();
    if (!msg.stateMatches(TAKEN)) return false;
    out.status = (msg.success != 0) ? tt::domain::ManageMemoryStatus::SUCCESS
                                    : tt::domain::ManageMemoryStatus::FAILURE;
    out.task_id =
        tt::domain::TaskID::ipcDeserialize(msg.taskId, sizeof(msg.taskId));
    out.memory_locations.resize(msg.num_destinations);
    for (uint32_t i = 0; i < msg.num_destinations; ++i) {
      out.memory_locations[i].dram_address = msg.destinations[i].dram_address;
      out.memory_locations[i].semaphore_address =
          msg.destinations[i].semaphore_address;
    }
    msg.switchState(FREE);
    advanceCurrent();
    return true;
  }

  std::string name;

 private:
  SlotType& acquireSlot() { return messages[current]; }
  void advanceCurrent() { current = (current + 1) % SHM_SLOTS; }

  bool create_;
  bool owner_;
  void* memPointer = nullptr;
  uint64_t current = 0;
  std::span<SlotType> messages;
};

using PrefillSharedMemory = SharedMemory<Message<PREFILL_MAX_TOKEN_IDS>>;
using DecodeSharedMemory = SharedMemory<Message<DECODE_MAX_TOKEN_IDS>>;
using MemoryRequestQueue = SharedMemory<MemoryRequestSlot>;
using MemoryResultQueue = SharedMemory<MemoryResultSlot>;

inline constexpr const char* k_memory_request_shm_name = "/tt_mem_requests";
inline constexpr const char* k_memory_result_shm_name = "/tt_mem_results";

}  // namespace sp_pipeline
