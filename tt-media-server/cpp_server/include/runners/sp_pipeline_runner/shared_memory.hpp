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
    uint32_t maxTokens;
    uint32_t numTokenIds;
    char taskId[36];
    uint64_t tokenIds[MaxTokenIds];

    static constexpr int kMessageSize = 48 + MaxTokenIds * sizeof(uint64_t);
    static constexpr int kTotalSize = SHM_SLOTS * kMessageSize;
    
    bool stateMatches(SlotState state) {
        return this->state.load(std::memory_order_acquire) == state;
    }
    
    void switchState(SlotState state) {
        this->state.store(state, std::memory_order_release);
    }
};

static_assert(sizeof(Message<PREFILL_MAX_TOKEN_IDS>) == Message<PREFILL_MAX_TOKEN_IDS>::kMessageSize);
static_assert(sizeof(Message<DECODE_MAX_TOKEN_IDS>) == Message<DECODE_MAX_TOKEN_IDS>::kMessageSize);

struct ReadResult {
    std::string taskId;
    uint32_t maxTokens;
    std::vector<int64_t> tokenIds;
};

template<int MaxTokenIds>
class SharedMemory {
public:
    using Msg = Message<MaxTokenIds>;

    explicit SharedMemory(const std::string& name)
        : name(name[0] == '/' ? name : "/" + name) {}

    ~SharedMemory() {
        if (memPointer && memPointer != MAP_FAILED) {
            munmap(memPointer, Msg::kTotalSize);
        }
        shm_unlink(name.c_str());
    }

    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    void open() {
        int fd = shm_open(name.c_str(), O_RDWR, 0);
        if (fd < 0) {
            throw std::runtime_error("SharedMemory: unable to open shared memory: " + name);
        }

        memPointer = mmap(nullptr, Msg::kTotalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (memPointer == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("SharedMemory: mmap failed: " + name);
        }
        ::close(fd);

        std::memset(memPointer, 0, Msg::kTotalSize);
        messages = std::span<Msg>(static_cast<Msg*>(memPointer), SHM_SLOTS);
    }

    void write(const std::string& uuid, const std::vector<int64_t>& token_ids, uint32_t max_tokens) {
        if (static_cast<int>(token_ids.size()) > MaxTokenIds) {
            throw std::runtime_error(
                "SharedMemory::write: token count " + std::to_string(token_ids.size()) +
                " exceeds slot capacity " + std::to_string(MaxTokenIds));
        }

        auto& msg = acquireMsg();

        while (!msg.stateMatches(FREE)) {
            std::this_thread::yield();
        }

        msg.maxTokens = max_tokens;
        msg.numTokenIds = token_ids.size();
        std::memcpy(msg.taskId, uuid.data(), 36);
        std::memcpy(msg.tokenIds, token_ids.data(), token_ids.size() * sizeof(int64_t));

        msg.switchState(TAKEN);
        advanceCurrent();
    }

    bool try_read(ReadResult& out) {
        auto& msg = acquireMsg();

        if (!msg.stateMatches(TAKEN)) {
            return false;
        }

        out.taskId.assign(msg.taskId, 36);
        out.maxTokens = msg.maxTokens;
        out.tokenIds.assign(msg.tokenIds, msg.tokenIds + msg.numTokenIds);

        msg.switchState(FREE);
        advanceCurrent();
        return true;
    }
    std::string name;

private:
    Msg& acquireMsg() {
        return messages[current];
    }
    
    void advanceCurrent() {
        current = (current + 1) % SHM_SLOTS;
    }
    
    void* memPointer = nullptr;
    uint64_t current = 0;
    std::span<Msg> messages;
};

using PrefillSharedMemory = SharedMemory<PREFILL_MAX_TOKEN_IDS>;
using DecodeSharedMemory = SharedMemory<DECODE_MAX_TOKEN_IDS>;

}  // namespace sp_pipeline
