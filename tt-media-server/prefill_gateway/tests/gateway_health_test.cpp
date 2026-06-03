// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_health.hpp"

#include "gateway/prefill_registry.hpp"
#include "gtest/gtest.h"

namespace tt::gateway {
namespace {

TEST(GatewayHealthTest, ReadyWhenDecodeConnectedAndPrefillsHealthy) {
  PrefillRegistry registry;
  registry.preRegister("p1", nullptr);
  ASSERT_TRUE(registry.markRegistered("p1", 4));

  const GatewayHealthStatus status =
      buildGatewayHealthStatus(registry, "tcp", true);

  EXPECT_TRUE(status.ready);
  EXPECT_TRUE(status.error.empty());
  EXPECT_NE(status.livenessJson.find(R"("status":"alive")"), std::string::npos);
  EXPECT_NE(status.healthJson.find(R"("status":"healthy")"), std::string::npos);
  EXPECT_NE(status.livenessJson.find(R"("decode_connected":true)"),
            std::string::npos);
}

TEST(GatewayHealthTest, NotReadyWhenDecodeDisconnected) {
  PrefillRegistry registry;
  registry.preRegister("p1", nullptr);
  ASSERT_TRUE(registry.markRegistered("p1", 4));

  const GatewayHealthStatus status =
      buildGatewayHealthStatus(registry, "tcp", false);

  EXPECT_FALSE(status.ready);
  EXPECT_EQ(status.error, "decode not connected");
  EXPECT_NE(status.livenessJson.find(R"("status":"alive")"), std::string::npos);
  EXPECT_NE(status.healthJson.find(R"("status":"unhealthy")"),
            std::string::npos);
  EXPECT_NE(status.healthJson.find(R"("error":"decode not connected")"),
            std::string::npos);
}

TEST(GatewayHealthTest, NotReadyWhenNoHealthyPrefills) {
  PrefillRegistry registry;
  registry.preRegister("p1", nullptr);

  const GatewayHealthStatus status =
      buildGatewayHealthStatus(registry, "tcp", true);

  EXPECT_FALSE(status.ready);
  EXPECT_EQ(status.error, "no healthy prefills");
}

TEST(GatewayHealthTest, NotReadyWhenNoPrefillsAcceptingTasks) {
  PrefillRegistry registry;
  registry.preRegister("p1", nullptr);
  ASSERT_TRUE(registry.markRegistered("p1", 4));
  registry.setAcceptingTasks("p1", false);

  const GatewayHealthStatus status =
      buildGatewayHealthStatus(registry, "tcp", true);

  EXPECT_FALSE(status.ready);
  EXPECT_EQ(status.error, "no prefills accepting tasks");
}

}  // namespace
}  // namespace tt::gateway
