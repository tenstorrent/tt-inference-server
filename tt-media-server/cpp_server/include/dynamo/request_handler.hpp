// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "dynamo/transport_protocol.hpp"

namespace trantor {
class EventLoopThreadPool;
}

namespace tt::services {
class DisaggregationService;
class LLMPipeline;
}  // namespace tt::services

namespace tt::dynamo {

class DynamoRequestHandler {
 public:
  DynamoRequestHandler(
      std::shared_ptr<services::LLMPipeline> pipeline,
      std::shared_ptr<services::DisaggregationService> disaggregation,
      trantor::EventLoopThreadPool* loopPool);

  void handle(const GenerateRequest& dynReq,
              const TcpStreamConnectionInfo& connInfo);

 private:
  std::shared_ptr<services::LLMPipeline> pipeline_;
  std::shared_ptr<services::DisaggregationService> disaggregation_;
  trantor::EventLoopThreadPool* loop_pool_;
};

}  // namespace tt::dynamo
