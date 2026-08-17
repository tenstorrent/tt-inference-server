// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>

namespace tt::worker {

/** Liveness / status entry for one worker process (see
 * WorkerManager::getWorkerInfo). */
struct WorkerInfo {
  std::string worker_id;
  bool is_ready = false;
  bool is_alive = true;
  pid_t pid = -1;
};

}  // namespace tt::worker
