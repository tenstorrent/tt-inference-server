// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_metrics.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

namespace tt::gateway {
namespace {

TEST(GatewayMetricsTest, RendersGatewayCountersAndGauges) {
  auto& metrics = GatewayMetrics::instance();
  metrics.resetForTests();

  metrics.recordRoutingDecision("prefix_match");
  metrics.observePrefixMatchDepth(3);
  metrics.setRoutingTableSize(7);
  metrics.recordRequestCompleted("prefill-a", "success",
                                 std::chrono::milliseconds(125));
  metrics.recordRequestFailed("timeout");
  metrics.recordTimeout("prefill-a");
  metrics.recordCancel(true);
  metrics.setDecodeConnected(true);
  metrics.setPrefillSnapshots(
      {GatewayPrefillMetricSnapshot{"prefill-a", true, true, 2, 11, 1.5}});

  const std::string text = metrics.renderText();

  EXPECT_NE(text.find("tt_gateway_routing_decisions_total"), std::string::npos);
  EXPECT_NE(text.find("reason=\"prefix_match\""), std::string::npos);
  EXPECT_NE(text.find("tt_prefill_completed_total"), std::string::npos);
  EXPECT_NE(text.find("server_id=\"prefill-a\""), std::string::npos);
  EXPECT_NE(text.find("outcome=\"success\""), std::string::npos);
  EXPECT_NE(text.find("tt_prefill_latency_seconds"), std::string::npos);
  EXPECT_NE(text.find("tt_prefill_inflight"), std::string::npos);
  EXPECT_NE(text.find("tt_gateway_decode_connected"), std::string::npos);
}

}  // namespace
}  // namespace tt::gateway
