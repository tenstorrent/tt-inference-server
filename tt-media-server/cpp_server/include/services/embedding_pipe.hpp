// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

/**
 * Length-prefixed pipe framing shared by the embedding parent process and its
 * forked workers. Every message is [len:u32][payload]; the READY sentinel is
 * the one special payload, sent by a worker after successful warmup.
 */
namespace tt::services::embedding_detail {

/// Sent by a worker child over its response pipe once warmup succeeds, so the
/// parent can distinguish "forked" from "actually able to serve requests".
inline constexpr char WORKER_READY_SENTINEL[] = "READY";

/** Length-prefixed pipe write: [len:u32][payload]. Returns false on failure.
 */
bool pipeWrite(int fd, const void* data, size_t len);

/** Length-prefixed pipe read. Returns an empty vector on failure or when the
 * length prefix exceeds EMBEDDING_MAX_PIPE_BYTES. */
std::vector<uint8_t> pipeReadBinary(int fd);

/** Length-prefixed pipe read into a string. Returns an empty string on
 * failure. */
std::string pipeReadString(int fd);

/** True when the payload is exactly the READY sentinel. */
bool isReadySentinel(const std::vector<uint8_t>& payload);

}  // namespace tt::services::embedding_detail
