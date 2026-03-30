// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include "domain/task_id.hpp"

namespace tt::domain {

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
  TaskID task_id;
  MemoryManagementAction action{MemoryManagementAction::ALLOCATE};
  std::int32_t input_seq_len{0};
  KvMemoryLayout memory_layout{KvMemoryLayout::Paged};
  std::vector<std::int32_t> slot_ids;

  void serialize(std::ostream& os) const {
    auto tid_buf = task_id.ipcSerialize();
    os.write(tid_buf.data(), static_cast<std::streamsize>(tid_buf.size()));
    auto a = static_cast<std::uint8_t>(action);
    os.write(reinterpret_cast<const char*>(&a), sizeof(a));
    os.write(reinterpret_cast<const char*>(&input_seq_len),
             sizeof(input_seq_len));
    auto ml = static_cast<std::uint8_t>(memory_layout);
    os.write(reinterpret_cast<const char*>(&ml), sizeof(ml));
    std::uint32_t n = static_cast<std::uint32_t>(slot_ids.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (std::int32_t id : slot_ids) {
      os.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
  }

  static ManageMemoryTask deserialize(std::istream& is) {
    ManageMemoryTask task;
    char tid_buf[TaskID::K_SERIALIZED_SIZE];
    is.read(tid_buf, TaskID::K_SERIALIZED_SIZE);
    task.task_id = TaskID::ipcDeserialize(tid_buf, TaskID::K_SERIALIZED_SIZE);
    std::uint8_t a = 0;
    is.read(reinterpret_cast<char*>(&a), sizeof(a));
    task.action = static_cast<MemoryManagementAction>(a);
    is.read(reinterpret_cast<char*>(&task.input_seq_len),
            sizeof(task.input_seq_len));
    std::uint8_t ml = 0;
    is.read(reinterpret_cast<char*>(&ml), sizeof(ml));
    task.memory_layout = static_cast<KvMemoryLayout>(ml);
    std::uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    task.slot_ids.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      is.read(reinterpret_cast<char*>(&task.slot_ids[i]),
              sizeof(std::int32_t));
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
  TaskID task_id;
  ManageMemoryStatus status{ManageMemoryStatus::FAILURE};
  std::vector<std::int32_t> slot_ids;

  void serialize(std::ostream& os) const {
    auto tid_buf = task_id.ipcSerialize();
    os.write(tid_buf.data(), static_cast<std::streamsize>(tid_buf.size()));
    auto s = static_cast<std::uint8_t>(status);
    os.write(reinterpret_cast<const char*>(&s), sizeof(s));
    std::uint32_t n = static_cast<std::uint32_t>(slot_ids.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (std::int32_t id : slot_ids) {
      os.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
  }

  static ManageMemoryResult deserialize(std::istream& is) {
    ManageMemoryResult result;
    char tid_buf[TaskID::K_SERIALIZED_SIZE];
    is.read(tid_buf, TaskID::K_SERIALIZED_SIZE);
    result.task_id = TaskID::ipcDeserialize(tid_buf, TaskID::K_SERIALIZED_SIZE);
    std::uint8_t s = 0;
    is.read(reinterpret_cast<char*>(&s), sizeof(s));
    result.status = static_cast<ManageMemoryStatus>(s);
    std::uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    result.slot_ids.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      is.read(reinterpret_cast<char*>(&result.slot_ids[i]),
              sizeof(std::int32_t));
    }
    return result;
  }
};

}  // namespace tt::domain
