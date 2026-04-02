// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <vector>

namespace tt::domain {

constexpr uint32_t INVALID_SLOT_ID = std::numeric_limits<uint32_t>::max();

enum class MemoryManagementAction : std::uint8_t {
  ALLOCATE = 0,
  DEALLOCATE = 1,
  MOVE = 2,
};

enum class KvMemoryLayout : std::uint8_t {
  Paged = 0,
  PerLayer = 1,
};

struct ManageMemoryTask {
  uint32_t taskId;
  MemoryManagementAction action{MemoryManagementAction::ALLOCATE};
  std::uint32_t inputSeqLen{0};
  KvMemoryLayout memoryLayout{KvMemoryLayout::Paged};
  std::vector<std::uint32_t> slotIds;

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
    auto a = static_cast<std::uint8_t>(action);
    os.write(reinterpret_cast<const char*>(&a), sizeof(a));
    os.write(reinterpret_cast<const char*>(&inputSeqLen), sizeof(inputSeqLen));
    auto ml = static_cast<std::uint8_t>(memoryLayout);
    os.write(reinterpret_cast<const char*>(&ml), sizeof(ml));
    std::uint32_t n = static_cast<std::uint32_t>(slotIds.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (std::uint32_t id : slotIds) {
      os.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
  }

  static ManageMemoryTask deserialize(std::istream& is) {
    ManageMemoryTask task;
    is.read(reinterpret_cast<char*>(&task.taskId), sizeof(task.taskId));
    std::uint8_t a = 0;
    is.read(reinterpret_cast<char*>(&a), sizeof(a));
    task.action = static_cast<MemoryManagementAction>(a);
    is.read(reinterpret_cast<char*>(&task.inputSeqLen),
            sizeof(task.inputSeqLen));
    std::uint8_t ml = 0;
    is.read(reinterpret_cast<char*>(&ml), sizeof(ml));
    task.memoryLayout = static_cast<KvMemoryLayout>(ml);
    std::uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    task.slotIds.resize(n, INVALID_SLOT_ID);
    for (std::uint32_t i = 0; i < n; ++i) {
      is.read(reinterpret_cast<char*>(&task.slotIds[i]), sizeof(std::uint32_t));
    }
    return task;
  }
};

enum class ManageMemoryStatus : std::uint8_t {
  SUCCESS = 0,
  FAILURE = 1,
  WAITING = 2,
};

struct ManageMemoryResult {
  uint32_t taskId;
  ManageMemoryStatus status{ManageMemoryStatus::FAILURE};
  std::vector<std::uint32_t> slotIds;

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
    auto s = static_cast<std::uint8_t>(status);
    os.write(reinterpret_cast<const char*>(&s), sizeof(s));
    std::uint32_t n = static_cast<std::uint32_t>(slotIds.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (std::uint32_t id : slotIds) {
      os.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
  }

  static ManageMemoryResult deserialize(std::istream& is) {
    ManageMemoryResult result;
    is.read(reinterpret_cast<char*>(&result.taskId), sizeof(result.taskId));
    std::uint8_t s = 0;
    is.read(reinterpret_cast<char*>(&s), sizeof(s));
    result.status = static_cast<ManageMemoryStatus>(s);
    std::uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    result.slotIds.resize(n, INVALID_SLOT_ID);
    for (std::uint32_t i = 0; i < n; ++i) {
      is.read(reinterpret_cast<char*>(&result.slotIds[i]),
              sizeof(std::uint32_t));
    }
    return result;
  }
};

}  // namespace tt::domain
