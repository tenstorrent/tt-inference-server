#pragma once

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
struct SharedMemoryLayout {
    // SLOTS = 1 is sufficient while max_num_seqs = 1 in config.hpp.
    // Increase if max_num_seqs grows.
    static constexpr int SLOTS = 1;
    static constexpr int MAX_PAYLOAD_SIZE = 128000 * sizeof(uint64_t);
    static constexpr int MESSAGE_SIZE = 64 + MAX_PAYLOAD_SIZE;
    static constexpr int TOTAL_SIZE = SLOTS * MESSAGE_SIZE;
};

enum SlotState { FREE, TAKEN };

struct Message {
    // Header
    std::atomic<int32_t> state;
    uint32_t max_tokens;
    uint32_t num_tokens;
    char task_id[36];
    uint8_t reserved[16];

    uint64_t tokens[SharedMemoryLayout::MAX_PAYLOAD_SIZE / sizeof(uint64_t)]; 
};

static_assert(sizeof(Message) == SharedMemoryLayout::MESSAGE_SIZE, "Message size mismatch");

struct ReadResult {
    char task_id[36];
    uint64_t token;
};

class SharedMemory {
public:
    explicit SharedMemory(const std::string& name);
    ~SharedMemory();

    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    void open();

    void write(const std::string& task_id,
               const std::vector<int64_t>& token_ids,
               uint32_t max_tokens);

    bool try_read(ReadResult& out);

    const std::string& getName() const { return name; }

private:
    std::string name;
    void* memPointer = nullptr;
    uint64_t current = 0;
    std::span<Message> slots;
};
