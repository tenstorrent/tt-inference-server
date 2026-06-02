// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <ostream>
namespace tt::domain {

enum class MemoryManagementAction : std::uint8_t {
  ALLOCATE = 0,
  DEALLOCATE = 1,
  MOVE = 2,
};

enum class KvMemoryLayout : std::uint8_t {
  PAGED = 0,
  PER_LAYER = 1,
};

struct ManageMemoryTask {
  uint32_t taskId;
  MemoryManagementAction action{MemoryManagementAction::ALLOCATE};
  KvMemoryLayout memoryLayout{KvMemoryLayout::PAGED};
  uint32_t slotId{};
  std::optional<uint32_t> slotIdToCopyFrom;

  ManageMemoryTask() = default;
  ManageMemoryTask(uint32_t id, MemoryManagementAction a)
      : taskId(id), action(a) {}

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
    auto a = static_cast<std::uint8_t>(action);
    os.write(reinterpret_cast<const char*>(&a), sizeof(a));
    auto ml = static_cast<std::uint8_t>(memoryLayout);
    os.write(reinterpret_cast<const char*>(&ml), sizeof(ml));
    os.write(reinterpret_cast<const char*>(&slotId), sizeof(slotId));
    uint8_t hasCopyFrom = slotIdToCopyFrom.has_value() ? 1 : 0;
    os.write(reinterpret_cast<const char*>(&hasCopyFrom), sizeof(hasCopyFrom));
    if (hasCopyFrom) {
      uint32_t copyFrom = *slotIdToCopyFrom;
      os.write(reinterpret_cast<const char*>(&copyFrom), sizeof(copyFrom));
    }
  }

  static ManageMemoryTask deserialize(std::istream& is) {
    ManageMemoryTask task;
    is.read(reinterpret_cast<char*>(&task.taskId), sizeof(task.taskId));
    std::uint8_t a = 0;
    is.read(reinterpret_cast<char*>(&a), sizeof(a));
    task.action = static_cast<MemoryManagementAction>(a);
    std::uint8_t ml = 0;
    is.read(reinterpret_cast<char*>(&ml), sizeof(ml));
    task.memoryLayout = static_cast<KvMemoryLayout>(ml);
    is.read(reinterpret_cast<char*>(&task.slotId), sizeof(task.slotId));
    uint8_t hasCopyFrom = 0;
    is.read(reinterpret_cast<char*>(&hasCopyFrom), sizeof(hasCopyFrom));
    if (hasCopyFrom) {
      uint32_t copyFrom = 0;
      is.read(reinterpret_cast<char*>(&copyFrom), sizeof(copyFrom));
      task.slotIdToCopyFrom = copyFrom;
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
  uint32_t slotId;

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&taskId), sizeof(taskId));
    auto s = static_cast<std::uint8_t>(status);
    os.write(reinterpret_cast<const char*>(&s), sizeof(s));
    os.write(reinterpret_cast<const char*>(&slotId), sizeof(slotId));
  }

  static ManageMemoryResult deserialize(std::istream& is) {
    ManageMemoryResult result;
    is.read(reinterpret_cast<char*>(&result.taskId), sizeof(result.taskId));
    std::uint8_t s = 0;
    is.read(reinterpret_cast<char*>(&s), sizeof(s));
    result.status = static_cast<ManageMemoryStatus>(s);
    is.read(reinterpret_cast<char*>(&result.slotId), sizeof(result.slotId));
    return result;
  }
};

}  // namespace tt::domain
