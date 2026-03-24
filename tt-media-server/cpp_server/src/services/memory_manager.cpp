// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

#include <utility>

namespace tt::services {

namespace {

using tt::domain::KvDestination;
using tt::domain::KvMemoryLayout;
using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;

ManageMemoryResult makeFailedResult(const ManageMemoryTask& task) {
  return ManageMemoryResult{
      .task_id = task.task_id, .success = false, .memory_locations = {}};
}

ManageMemoryResult makeSuccessResult(
    const ManageMemoryTask& task, std::vector<KvDestination> memoryLocations) {
  return ManageMemoryResult{.task_id = task.task_id,
                            .success = true,
                            .memory_locations = std::move(memoryLocations)};
}

bool allocateKv(const ManageMemoryTask& /*task*/,
                std::vector<KvDestination>& /*out*/) {
  // TODO(ttnn): Size and fill from device.
  return true;
}

void deallocateKv(const std::vector<KvDestination>& /*locations*/) {
  // TODO(ttnn): Release KV via ttnn / device.
}

}  // namespace

ManageMemoryResult MemoryManager::handle_task(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    return makeFailedResult(task);
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        if (reservations_.contains(task.task_id.id)) {
          return makeFailedResult(task);
        }
        if (task.memory_layout == KvMemoryLayout::PerLayer) {
          return makeFailedResult(task);
        }
        if (task.input_seq_len < 0) {
          return makeFailedResult(task);
        }
        reservations_.emplace(
            task.task_id.id,
            Reservation{.layout = KvMemoryLayout::Paged, .locations = {}});
      }

      std::vector<KvDestination> locations;
      if (!allocateKv(task, locations)) {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        reservations_.erase(task.task_id.id);
        return makeFailedResult(task);
      }

      {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        reservations_[task.task_id.id].locations = locations;
      }
      return makeSuccessResult(task, std::move(locations));
    }
    case MemoryManagementAction::DEALLOCATE: {
      std::vector<KvDestination> locations;
      {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        auto it = reservations_.find(task.task_id.id);
        if (it == reservations_.end()) {
          return makeFailedResult(task);
        }
        if (it->second.layout != task.memory_layout) {
          return makeFailedResult(task);
        }
        locations = std::move(it->second.locations);
        reservations_.erase(it);
      }

      deallocateKv(locations);
      return makeSuccessResult(task, {});
    }
    default:
      return makeFailedResult(task);
  }
}

}  // namespace tt::services
