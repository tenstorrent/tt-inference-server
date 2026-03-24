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

ManageMemoryResult make_failed_result(const ManageMemoryTask& task) {
  return ManageMemoryResult{
      .task_id = task.task_id, .success = false, .memory_locations = {}};
}

ManageMemoryResult make_success_result(
    const ManageMemoryTask& task, std::vector<KvDestination> memory_locations) {
  return ManageMemoryResult{.task_id = task.task_id,
                            .success = true,
                            .memory_locations = std::move(memory_locations)};
}

bool allocate_kv(const ManageMemoryTask& /*task*/,
                 std::vector<KvDestination>& /*out*/) {
  // TODO(ttnn): Size and fill from device.
  return true;
}

void deallocate_kv(const std::vector<KvDestination>& /*locations*/) {
  // TODO(ttnn): Release KV via ttnn / device.
}

}  // namespace

ManageMemoryResult MemoryManager::handle_task(const ManageMemoryTask& task) {
  if (task.action == MemoryManagementAction::MOVE) {
    return make_failed_result(task);
  }

  switch (task.action) {
    case MemoryManagementAction::ALLOCATE: {
      {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        if (reservations_.contains(task.task_id.id)) {
          return make_failed_result(task);
        }
        if (task.memory_layout == KvMemoryLayout::PerLayer) {
          return make_failed_result(task);
        }
        if (task.input_seq_len < 0) {
          return make_failed_result(task);
        }
        reservations_.emplace(
            task.task_id.id,
            Reservation{.layout = KvMemoryLayout::Paged, .locations = {}});
      }

      std::vector<KvDestination> locations;
      if (!allocate_kv(task, locations)) {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        reservations_.erase(task.task_id.id);
        return make_failed_result(task);
      }

      {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        reservations_[task.task_id.id].locations = locations;
      }
      return make_success_result(task, std::move(locations));
    }
    case MemoryManagementAction::DEALLOCATE: {
      std::vector<KvDestination> locations;
      {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        auto it = reservations_.find(task.task_id.id);
        if (it == reservations_.end()) {
          return make_failed_result(task);
        }
        if (it->second.layout != task.memory_layout) {
          return make_failed_result(task);
        }
        locations = std::move(it->second.locations);
        reservations_.erase(it);
      }

      deallocate_kv(locations);
      return make_success_result(task, {});
    }
    default:
      return make_failed_result(task);
  }
}

}  // namespace tt::services
