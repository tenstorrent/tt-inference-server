// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace tt::domain {

struct BaseRequest {
  uint32_t task_id;

  // Per-process atomic `task_id` resets on restart and collides across nodes,
  // so it can't be used to follow one user request across decode HTTP, decode
  // worker, prefill HTTP, prefill worker and the inter-server socket. The
  // `trace_id` is set once at HTTP entry — either copied from the inbound
  // `X-Request-Id` header or generated via TraceIdGenerator — and is
  // propagated through every downstream message and log line for that
  // request.
  std::string trace_id;

  explicit BaseRequest(uint32_t taskId) : task_id(taskId) {}
  BaseRequest(uint32_t taskId, std::string traceId)
      : task_id(taskId), trace_id(std::move(traceId)) {}
};

}  // namespace tt::domain
