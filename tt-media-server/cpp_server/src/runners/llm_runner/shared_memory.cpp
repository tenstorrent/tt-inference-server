#include "runners/llm_runner/shared_memory.hpp"

#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>

SharedMemory::SharedMemory(const std::string& name)
    : name(name + std::to_string(getpid())) {}

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
    std::memcpy(slot.payload, uuid.data(), 36);
    std::memcpy(slot.payload + 36, &token, 8);
    slot.state.store(SlotState::TAKEN, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
}

std::vector<uint8_t> SharedMemory::read() {
    std::vector<uint8_t> bytes(44);
    auto& slot = slots[current];
    while (slot.state.load(std::memory_order_acquire) != SlotState::TAKEN) {
        std::this_thread::yield();
    }
    std::memcpy(bytes.data(), slot.payload, 44);
    slot.state.store(SlotState::FREE, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
    return bytes;
}

bool SharedMemory::try_read(std::vector<uint8_t>& out) {
    auto& slot = slots[current];
    if (slot.state.load(std::memory_order_acquire) != SlotState::TAKEN) {
        return false;
    }
    out.resize(44);
    std::memcpy(out.data(), slot.payload, 44);
    slot.state.store(SlotState::FREE, std::memory_order_release);
    current = (current + 1) % SharedMemoryConfig::SLOTS;
    return true;
}
