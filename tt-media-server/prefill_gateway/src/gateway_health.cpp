// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/gateway_health.hpp"

#include <sstream>

#include "gateway/prefill_registry.hpp"

namespace tt::gateway {
namespace {

std::string buildHealthJsonBody(std::string_view status, std::string_view error,
                                std::string_view transport,
                                size_t registeredPrefills,
                                size_t healthyPrefills,
                                size_t acceptingPrefills,
                                bool decodeConnected) {
  std::ostringstream out;
  out << "{\"status\":\"" << status << "\"";
  if (!error.empty()) {
    out << ",\"error\":\"" << error << "\"";
  }
  out << ",\"transport\":\"" << transport << "\""
      << ",\"registered_prefills\":" << registeredPrefills
      << ",\"healthy_prefills\":" << healthyPrefills
      << ",\"accepting_prefills\":" << acceptingPrefills
      << ",\"decode_connected\":" << (decodeConnected ? "true" : "false")
      << "}\n";
  return out.str();
}

}  // namespace

GatewayHealthStatus buildGatewayHealthStatus(const PrefillRegistry& registry,
                                             std::string_view transport,
                                             bool decodeConnected) {
  const auto prefills = registry.snapshot();
  size_t healthyPrefills = 0;
  size_t acceptingPrefills = 0;
  for (const auto& prefill : prefills) {
    if (prefill.healthy) ++healthyPrefills;
    if (prefill.accepting_tasks) ++acceptingPrefills;
  }

  GatewayHealthStatus result;
  if (!decodeConnected) {
    result.error = "decode not connected";
  } else if (healthyPrefills == 0) {
    result.error = "no healthy prefills";
  } else if (acceptingPrefills == 0) {
    result.error = "no prefills accepting tasks";
  }
  result.ready = result.error.empty();

  result.livenessJson =
      buildHealthJsonBody("alive", "", transport, prefills.size(),
                          healthyPrefills, acceptingPrefills, decodeConnected);
  result.healthJson = buildHealthJsonBody(
      result.ready ? "healthy" : "unhealthy", result.error, transport,
      prefills.size(), healthyPrefills, acceptingPrefills, decodeConnected);
  return result;
}

}  // namespace tt::gateway
