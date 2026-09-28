// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// Pipe framing between the embedding parent and its forked workers.
/// Every message is [len:u32][payload].
namespace tt::services::embedding_detail {

/// Sent by a worker once warmup succeeds.
inline constexpr char WORKER_READY_SENTINEL[] = "READY";

/// Writes [len:u32][payload]; false on failure.
bool pipeWrite(int fd, const void* data, size_t len);

/// Reads one framed message; empty on failure or oversized length.
std::vector<uint8_t> pipeReadBinary(int fd);

/// String variant of pipeReadBinary.
std::string pipeReadString(int fd);

/// True when the payload is exactly the READY sentinel.
bool isReadySentinel(const std::vector<uint8_t>& payload);

}  // namespace tt::services::embedding_detail
