// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>

namespace tt::worker {

/** Liveness / status entry for one worker process (see
 * WorkerManager::getWorkerInfo). */
struct WorkerInfo {
  std::string worker_id;
  bool is_ready;
};

}  // namespace tt::worker
