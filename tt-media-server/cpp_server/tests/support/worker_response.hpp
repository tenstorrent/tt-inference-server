// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Fluent builder for the token stream a mock worker pushes onto the result
// queue. Counterpart to ChatRequest on the request side:
//
//   WorkerResponse(taskId)
//       .token(42)
//       .token(43)
//       .finalize()                   // empty terminator with FLAG_FINAL
//       .sendTo(server_->resultQueue());
//
// token_index is auto-assigned in push order. Flags can be set explicitly
// via finalize() / errorFinal() / done() or per-token via tokenWithFlags().

#pragma once

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/result_queue.hpp"

namespace tt::test {

class WorkerResponse {
 public:
  explicit WorkerResponse(uint32_t taskId) : taskId_(taskId) {}

  WorkerResponse& token(uint64_t tokenId) { return tokenWithFlags(tokenId, 0); }

  WorkerResponse& tokens(std::initializer_list<uint64_t> ids) {
    for (uint64_t id : ids) token(id);
    return *this;
  }

  WorkerResponse& tokenWithFlags(uint64_t tokenId, uint32_t flags) {
    tokens_.push_back({tokenId, flags});
    return *this;
  }

  // Append an empty terminator with FLAG_FINAL — this is what production
  // workers send when generation completes normally.
  WorkerResponse& finalize() {
    return tokenWithFlags(0, ipc::SharedToken::FLAG_FINAL);
  }

  // Append an empty terminator with FLAG_FINAL | FLAG_ERROR — exercises
  // the controller's error path.
  WorkerResponse& errorFinal() {
    return tokenWithFlags(
        0, ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR);
  }

  void sendTo(ipc::BoostIpcResultQueue& queue) const {
    uint32_t idx = 0;
    for (const auto& [tokenId, flags] : tokens_) {
      ipc::SharedToken tok{};
      tok.task_id = taskId_;
      tok.token_index = idx++;
      tok.token_id = tokenId;
      tok.flags = flags;
      queue.push(tok);
    }
  }

 private:
  uint32_t taskId_;
  std::vector<std::pair<uint64_t, uint32_t>> tokens_;
};

}  // namespace tt::test
