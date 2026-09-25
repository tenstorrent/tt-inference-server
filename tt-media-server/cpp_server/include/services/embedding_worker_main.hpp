// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

namespace tt::services::embedding_detail {

/**
 * Child-process entry point, run right after fork(): builds the runner,
 * warms it up, sends READY, then serves batches until pipe EOF. Never
 * returns. The only embedding code that executes in the child process.
 */
[[noreturn]] void workerProcessMain(int workerId, int readFd, int writeFd);

}  // namespace tt::services::embedding_detail
