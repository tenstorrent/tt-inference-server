// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <string_view>

namespace tt::gateway {

class PrefillRegistry;

struct GatewayHealthStatus {
  std::string livenessJson;
  std::string healthJson;
  bool ready = false;
  std::string error;
};

GatewayHealthStatus buildGatewayHealthStatus(const PrefillRegistry& registry,
                                             std::string_view transport,
                                             bool decodeConnected);

}  // namespace tt::gateway
