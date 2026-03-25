// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

#include <utility>

namespace tt::services {

namespace {

using tt::domain::KvDestination;
using tt::domain::KvMemoryLayout;
using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;

ManageMemoryResult makeResult(const ManageMemoryTask& task,
                              ManageMemoryStatus status,
                              std::vector<KvDestination> locations = {}) {
  return ManageMemoryResult{.task_id = task.task_id,
                            .status = status,
                            .memory_locations = std::move(locations)};
}

ManageMemoryStatus allocateKv(const ManageMemoryTask& /*task*/,
                              std::vector<KvDestination>& /*out*/) {
  // TODO(ttnn): Size and fill from device. Return WAITING when full.
  return ManageMemoryStatus::SUCCESS;
}

void deallocateKv(const std::vector<KvDestination>& /*locations*/) {
  // TODO(ttnn): Release KV via ttnn / device.
}

}  // namespace

ManageMemoryResult MemoryManager::handle_task(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    return makeResult(task, ManageMemoryStatus::FAILURE);
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      {
        std::lock_guard<std::mutex> lock(reservationMutex);
        if (reservations.contains(task.task_id.id)) {
          return makeResult(task, ManageMemoryStatus::FAILURE);
        }
        if (task.memory_layout == KvMemoryLayout::PerLayer) {
          return makeResult(task, ManageMemoryStatus::FAILURE);
        }
        if (task.input_seq_len < 0) {
          return makeResult(task, ManageMemoryStatus::FAILURE);
        }
        reservations.emplace(
            task.task_id.id,
            Reservation{.layout = KvMemoryLayout::Paged, .locations = {}});
      }

      std::vector<KvDestination> locations;
      auto allocStatus = allocateKv(task, locations);
      if (allocStatus != ManageMemoryStatus::SUCCESS) {
        std::lock_guard<std::mutex> lock(reservationMutex);
        reservations.erase(task.task_id.id);
        return makeResult(task, allocStatus);
      }

      {
        std::lock_guard<std::mutex> lock(reservationMutex);
        reservations[task.task_id.id].locations = locations;
      }
      return makeResult(task, ManageMemoryStatus::SUCCESS,
                        std::move(locations));
    }
    case MemoryManagementAction::DEALLOCATE: {
      std::vector<KvDestination> locations;
      {
        std::lock_guard<std::mutex> lock(reservationMutex);
        auto it = reservations.find(task.task_id.id);
        if (it == reservations.end()) {
          return makeResult(task, ManageMemoryStatus::FAILURE);
        }
        if (it->second.layout != task.memory_layout) {
          return makeResult(task, ManageMemoryStatus::FAILURE);
        }
        locations = std::move(it->second.locations);
        reservations.erase(it);
      }

      deallocateKv(locations);
      return makeResult(task, ManageMemoryStatus::SUCCESS);
    }
    default:
      return makeResult(task, ManageMemoryStatus::FAILURE);
  }
}

}  // namespace tt::services
