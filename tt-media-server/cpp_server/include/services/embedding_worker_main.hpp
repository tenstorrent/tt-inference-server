// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

namespace tt::services::embedding_detail {

/**
 * Child-process entry point, run right after fork(): exports the worker's
 * environment, builds the runner, warms it up, sends the READY sentinel, then
 * serves length-prefixed JSON batches until pipe EOF. Never returns — exits
 * the worker process instead. This is the only embedding code that executes
 * in the child process, which is why it lives apart from the parent-side
 * service logic.
 */
[[noreturn]] void workerProcessMain(int workerId, int readFd, int writeFd);

}  // namespace tt::services::embedding_detail
