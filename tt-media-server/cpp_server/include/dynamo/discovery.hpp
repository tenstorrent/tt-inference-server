// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * File-based discovery registration for Dynamo.
 *
 * Writes the instance JSON and Model Descriptor Card (MDC) JSON files that
 * the Dynamo frontend watches at `${DYN_FILE_STORE}/v1/instances` and
 * `${DYN_FILE_STORE}/v1/mdc`, respectively. The frontend uses the MDC's
 * tokenizer paths to tokenize incoming requests before routing to this
 * worker, so the paths must match the tokenizer the worker actually uses.
 */

#include <cstdint>
#include <string>

namespace tt::dynamo {

struct DiscoveryConfig {
  std::string store_path = "/tmp/dynamo_store_kv";
  std::string namespace_name = "default";
  std::string component = "backend";
  std::string endpoint = "generate";
  std::string instance_id_hex;
  uint64_t instance_id = 0;
  /// "host:port/instance_id_hex/endpoint" — the address Dynamo dials to send
  /// us a request.
  std::string tcp_address;
  /// Slug shown in /v1/models (e.g. "deepseek-ai/DeepSeek-R1-0528").
  std::string model_name;
  /// Filesystem directory containing config.json + tokenizer.json +
  /// tokenizer_config.json.
  std::string model_path;
};

/**
 * Register this backend instance in the file-based discovery store. Must be
 * called after the Dynamo TCP listener binds (so the port is known). When
 * `quiet` is true, skip log output (used by keepalive re-registrations).
 */
void register_discovery(const DiscoveryConfig& config, bool quiet = false);

/// Remove the instance + MDC files this backend wrote.
void unregister_discovery(const DiscoveryConfig& config);

}  // namespace tt::dynamo
