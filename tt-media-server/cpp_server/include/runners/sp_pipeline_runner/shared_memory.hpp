// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sp_pipeline {

constexpr int SHM_SLOTS = 64;
constexpr int PREFILL_MAX_TOKEN_IDS = 131072;  // matches Config::MAX_INPUT_TOKENS (128k)
constexpr int DECODE_MAX_TOKEN_IDS = 1;

enum SlotState { FREE, TAKEN };

template<int MaxTokenIds>
struct Message {
    std::atomic<int32_t> state;
    uint32_t max_tokens;
    uint32_t num_token_ids;
    char task_id[36];
    uint64_t token_ids[MaxTokenIds];

    static constexpr int kMessageSize = 48 + MaxTokenIds * sizeof(uint64_t);
    static constexpr int kTotalSize = SHM_SLOTS * kMessageSize;
};

static_assert(sizeof(Message<PREFILL_MAX_TOKEN_IDS>) == Message<PREFILL_MAX_TOKEN_IDS>::kMessageSize);
static_assert(sizeof(Message<DECODE_MAX_TOKEN_IDS>) == Message<DECODE_MAX_TOKEN_IDS>::kMessageSize);

struct ReadResult {
    std::string task_id;
    uint32_t max_tokens;
    std::vector<int64_t> token_ids;
};

template<int MaxTokenIds>
class SharedMemory {
public:
    using Msg = Message<MaxTokenIds>;

    explicit SharedMemory(const std::string& name)
        : name_(name[0] == '/' ? name : "/" + name) {}

    ~SharedMemory() {
        if (mem_pointer_ && mem_pointer_ != MAP_FAILED) {
            munmap(mem_pointer_, Msg::kTotalSize);
        }
        shm_unlink(name_.c_str());
    }

    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    void open() {
        int fd = shm_open(name_.c_str(), O_RDWR, 0);
        if (fd < 0) {
            throw std::runtime_error("SharedMemory: unable to open shared memory: " + name_);
        }

        mem_pointer_ = mmap(nullptr, Msg::kTotalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mem_pointer_ == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("SharedMemory: mmap failed: " + name_);
        }
        ::close(fd);

        std::memset(mem_pointer_, 0, Msg::kTotalSize);
        slots_ = std::span<Msg>(static_cast<Msg*>(mem_pointer_), SHM_SLOTS);
    }

    void write(const std::string& uuid, const std::vector<int64_t>& token_ids, uint32_t max_tokens) {
        if (static_cast<int>(token_ids.size()) > MaxTokenIds) {
            throw std::runtime_error(
                "SharedMemory::write: token count " + std::to_string(token_ids.size()) +
                " exceeds slot capacity " + std::to_string(MaxTokenIds));
        }

        auto& slot = slots_[current_];
        while (slot.state.load(std::memory_order_acquire) != SlotState::FREE) {
            std::this_thread::yield();
        }

        slot.max_tokens = max_tokens;
        slot.num_token_ids = token_ids.size();
        std::memcpy(slot.task_id, uuid.data(), 36);
        std::memcpy(slot.token_ids, token_ids.data(), token_ids.size() * sizeof(int64_t));

        slot.state.store(SlotState::TAKEN, std::memory_order_release);
        current_ = (current_ + 1) % SHM_SLOTS;
    }

    bool try_read(ReadResult& out) {
        auto& slot = slots_[current_];
        if (slot.state.load(std::memory_order_acquire) != SlotState::TAKEN) {
            return false;
        }

        out.task_id.assign(slot.task_id, 36);
        out.max_tokens = slot.max_tokens;
        out.token_ids.assign(slot.token_ids, slot.token_ids + slot.num_token_ids);

        slot.state.store(SlotState::FREE, std::memory_order_release);
        current_ = (current_ + 1) % SHM_SLOTS;
        return true;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    void* mem_pointer_ = nullptr;
    uint64_t current_ = 0;
    std::span<Msg> slots_;
};

using PrefillSharedMemory = SharedMemory<PREFILL_MAX_TOKEN_IDS>;
using DecodeSharedMemory = SharedMemory<DECODE_MAX_TOKEN_IDS>;

}  // namespace sp_pipeline
