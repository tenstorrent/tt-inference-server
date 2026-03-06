#include "runners/llm_runner/shared_memory.hpp"

#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>

SharedMemory::SharedMemory(const std::string& name)
    : name(name[0] == '/' ? name : "/" + name) {}

SharedMemory::~SharedMemory() {
    if (memPointer && memPointer != MAP_FAILED) {
        munmap(memPointer, SharedMemoryConfig::TOTAL_SIZE);
    }
    shm_unlink(name.c_str());
}

void SharedMemory::open() {
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        throw std::runtime_error("SharedMemory: unable to open shared memory: " + name);
    }

    if (ftruncate(fd, SharedMemoryConfig::TOTAL_SIZE) == -1) {
        close(fd);
        throw std::runtime_error("SharedMemory: ftruncate failed: " + name);
    }
    if (fchmod(fd, 0666) == -1) {
        close(fd);
        throw std::runtime_error("SharedMemory: fchmod failed: " + name);
    }

    memPointer = mmap(
        nullptr,
        SharedMemoryConfig::TOTAL_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0
    );

    if (memPointer == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("SharedMemory: mmap failed: " + name);
    }
    close(fd);

    std::memset(memPointer, 0, SharedMemoryConfig::TOTAL_SIZE);
    slots = std::span<Message>(static_cast<Message*>(memPointer), SharedMemoryConfig::SLOTS);
}

void SharedMemory::write(const std::string& uuid, uint64_t token) {
    auto& slot = slots[current];
    while (slot.state.load(std::memory_order_acquire) != SlotState::FREE) {
        std::this_thread::yield();
    }

    // Write single token (decode phase)
    slot.payload_length = 44;  // 36 (uuid) + 8 (token)
    slot.max_tokens = 0;       // Not used for decode
    slot.num_token_ids = 1;    // Single token

    std::memcpy(slot.payload, uuid.data(), 36);
    std::memcpy(slot.payload + 36, &token, 8);

    slot.state.store(SlotState::TAKEN, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
}

void SharedMemory::write(const std::string& uuid, const std::vector<int64_t>& token_ids, uint32_t max_tokens) {
    auto& slot = slots[current];
    while (slot.state.load(std::memory_order_acquire) != SlotState::FREE) {
        std::this_thread::yield();
    }

    // Write multiple token_ids (prefill phase)
    size_t token_ids_bytes = token_ids.size() * sizeof(int64_t);
    slot.payload_length = 36 + token_ids_bytes;
    slot.max_tokens = max_tokens;
    slot.num_token_ids = token_ids.size();

    std::memcpy(slot.payload, uuid.data(), 36);
    std::memcpy(slot.payload + 36, token_ids.data(), token_ids_bytes);

    slot.state.store(SlotState::TAKEN, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
}

std::vector<uint8_t> SharedMemory::read() {
    auto& slot = slots[current];
    while (slot.state.load(std::memory_order_acquire) != SlotState::TAKEN) {
        std::this_thread::yield();
    }

    uint32_t payload_length = slot.payload_length;
    std::vector<uint8_t> bytes(payload_length + 8);  // +8 for max_tokens and num_token_ids

    // Pack: max_tokens (4) + num_token_ids (4) + payload (variable)
    std::memcpy(bytes.data(), &slot.max_tokens, 4);
    std::memcpy(bytes.data() + 4, &slot.num_token_ids, 4);
    std::memcpy(bytes.data() + 8, slot.payload, payload_length);

    slot.state.store(SlotState::FREE, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
    return bytes;
}

bool SharedMemory::try_read(std::vector<uint8_t>& out) {
    auto& slot = slots[current];
    if (slot.state.load(std::memory_order_acquire) != SlotState::TAKEN) {
        return false;
    }

    uint32_t payload_length = slot.payload_length;
    out.resize(payload_length + 8);  // +8 for max_tokens and num_token_ids

    // Pack: max_tokens (4) + num_token_ids (4) + payload (variable)
    std::memcpy(out.data(), &slot.max_tokens, 4);
    std::memcpy(out.data() + 4, &slot.num_token_ids, 4);
    std::memcpy(out.data() + 8, slot.payload, payload_length);

    slot.state.store(SlotState::FREE, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
    return true;
}
