#include "runners/deepseek_runner/shared_memory.hpp"

#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>


SharedMemory::SharedMemory(const std::string& name)
    : name(name[0] == '/' ? name : "/" + name) {}  // Ensure name starts with /

SharedMemory::~SharedMemory() {
    if (memPointer && memPointer != MAP_FAILED) {
        munmap(memPointer, SharedMemoryLayout::TOTAL_SIZE);
    }
    shm_unlink(name.c_str());
}

void SharedMemory::open() {
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        throw std::runtime_error("SharedMemory: unable to open shared memory: " + name);
    }

    if (ftruncate(fd, SharedMemoryLayout::TOTAL_SIZE) == -1) {
        close(fd);
        throw std::runtime_error("SharedMemory: ftruncate failed: " + name);
    }

    memPointer = mmap(
        nullptr,
        SharedMemoryLayout::TOTAL_SIZE,
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

    std::memset(memPointer, 0, SharedMemoryLayout::TOTAL_SIZE);
    slots = std::span<Message>(static_cast<Message*>(memPointer), SharedMemoryLayout::SLOTS);
}

void SharedMemory::write(const std::string& task_id,
                         const std::vector<int64_t>& token_ids,
                         uint32_t max_tokens) {
    auto& slot = slots[current];
    while (slot.state.load(std::memory_order_acquire) != SlotState::FREE) {
        std::this_thread::yield();
    }

    slot.max_tokens = max_tokens;
    slot.num_tokens = static_cast<uint32_t>(token_ids.size());
    std::memset(slot.task_id, 0, sizeof(slot.task_id));
    std::memcpy(slot.task_id, task_id.data(),
                std::min(task_id.size(), sizeof(slot.task_id)));
    std::memcpy(slot.tokens, token_ids.data(),
                token_ids.size() * sizeof(int64_t));

    slot.state.store(SlotState::TAKEN, std::memory_order_release);
    current = (current + 1) % SharedMemoryLayout::SLOTS;
}

bool SharedMemory::try_read(ReadResult& out) {
    auto& slot = slots[current];
    if (slot.state.load(std::memory_order_acquire) != SlotState::TAKEN) {
        return false;
    }

    std::memcpy(out.task_id, slot.task_id, sizeof(out.task_id));
    out.token = (slot.num_tokens > 0) ? slot.tokens[0] : 0;

    slot.state.store(SlotState::FREE, std::memory_order_release);
    current = (current + 1) % SharedMemoryLayout::SLOTS;
    return true;
}
