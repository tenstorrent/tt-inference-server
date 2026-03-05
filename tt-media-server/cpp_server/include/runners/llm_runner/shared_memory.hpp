#pragma once

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

struct SharedMemoryConfig {
    static constexpr int SLOTS = 1024;
    static constexpr int MAX_PAYLOAD_SIZE = 8192;  // Support long token sequences
    static constexpr int MESSAGE_SIZE = 8224;      // state(4) + pad(16) + payload_length(4) + max_tokens(4) + num_token_ids(4) + payload(8192)
    static constexpr int TOTAL_SIZE = SLOTS * MESSAGE_SIZE;
};

enum SlotState { FREE, TAKEN };

struct Message {
    std::atomic<int32_t> state;
    uint8_t pad[16];
    uint32_t payload_length;     // Actual bytes used in payload
    uint32_t max_tokens;         // Max tokens to generate
    uint32_t num_token_ids;      // Number of token IDs in payload
    uint8_t payload[SharedMemoryConfig::MAX_PAYLOAD_SIZE]; // Variable length data: task_id (36 bytes) + token_ids (variable)
};

static_assert(sizeof(Message) == SharedMemoryConfig::MESSAGE_SIZE, "Message size mismatch");

class SharedMemory {
public:
    explicit SharedMemory(const std::string& name);
    ~SharedMemory();

    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    void open();

    // Write single token (for decode phase)
    void write(const std::string& uuid, uint64_t token);

    // Write multiple token_ids (for prefill phase)
    void write(const std::string& uuid, const std::vector<int64_t>& token_ids, uint32_t max_tokens);

    // Blocking read - spins until a slot is TAKEN
    std::vector<uint8_t> read();

    // Non-blocking read - returns true if data was available
    bool try_read(std::vector<uint8_t>& out);

    const std::string& getName() const { return name; }

private:
    std::string name;
    void* memPointer = nullptr;
    uint64_t current = 0;
    std::span<Message> slots;
};
