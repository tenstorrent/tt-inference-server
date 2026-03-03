#pragma once

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

struct SharedMemoryConfig {
    static constexpr int SLOTS = 1024;
    static constexpr int MESSAGE_SIZE = 64;
    static constexpr int TOTAL_SIZE = SLOTS * MESSAGE_SIZE;
};

enum SlotState { FREE, TAKEN };

struct Message {
    std::atomic<int32_t> state;
    uint8_t pad[16];
    uint8_t payload[44]; // 36 bytes for uuid, 8 bytes for token
};

static_assert(sizeof(Message) == SharedMemoryConfig::MESSAGE_SIZE, "Message size mismatch");

class SharedMemory {
public:
    explicit SharedMemory(const std::string& name);
    ~SharedMemory();

    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    void open();
    void write(const std::string& uuid, uint64_t token);

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
