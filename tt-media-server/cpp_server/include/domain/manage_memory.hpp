// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
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

/** One KV slot for paged decode (per KV block), not per model layer. */
struct KvDestination {
  std::uint64_t dram_address{};
  std::uint64_t semaphore_address{};
};

struct ManageMemoryTask {
  TaskID task_id;
  MemoryManagementAction action{MemoryManagementAction::ALLOCATE};
  std::int32_t input_seq_len{0};
  KvMemoryLayout memory_layout{KvMemoryLayout::Paged};
};

enum class ManageMemoryStatus : std::uint8_t {
  SUCCESS = 0,
  FAILURE = 1,
  WAITING = 2,
};

struct ManageMemoryResult {
  TaskID task_id;
  ManageMemoryStatus status{ManageMemoryStatus::FAILURE};
  std::vector<KvDestination> memory_locations;
};

}  // namespace tt::domain
