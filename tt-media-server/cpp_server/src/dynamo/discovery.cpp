// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/discovery.hpp"

#include <json/json.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "config/settings.hpp"
#include "dynamo/etcd_client.hpp"
#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

constexpr int K_CONTEXT_LENGTH = 131072;

/// Frontend reads the checksum but doesn't validate it for routing — it's
/// used only for cache invalidation.
constexpr const char* K_BLAKE3_PLACEHOLDER =
    "blake3:0000000000000000000000000000000000000000000000000000000000000000";

/// Dynamo's slug validator rejects anything outside [a-z0-9_-]. HuggingFace
/// model ids have `/` and uppercase, so map them to `-` and lowercase.
std::string sanitizeSlug(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
        c == '_') {
      out += c;
    } else if (c >= 'A' && c <= 'Z') {
      out += static_cast<char>(c + ('a' - 'A'));
    } else {
      out += '-';
    }
  }
  return out;
}

std::string serializeJson(const Json::Value& v) {
  Json::StreamWriterBuilder b;
  b["indentation"] = "";
  b["emitUTF8"] = true;
  return Json::writeString(b, v);
}

/// Hierarchical key that Dynamo's KVStoreDiscovery uses for both file and
/// etcd backends: <namespace>/<component>/<endpoint>/<instance_id_hex>.
std::string instanceKey(const DiscoveryConfig& c) {
  return c.namespace_name + "/" + c.component + "/" + c.endpoint + "/" +
         c.instance_id_hex;
}

/// Build the instance JSON document the frontend dials over (transport.tcp).
Json::Value buildInstanceJson(const DiscoveryConfig& c) {
  Json::Value instance(Json::objectValue);
  instance["type"] = "Endpoint";
  instance["component"] = c.component;
  instance["endpoint"] = c.endpoint;
  instance["namespace"] = c.namespace_name;
  instance["instance_id"] = static_cast<Json::UInt64>(c.instance_id);

  Json::Value transport(Json::objectValue);
  transport["tcp"] = c.tcp_address;
  instance["transport"] = std::move(transport);
  instance["device_type"] = "cuda";
  return instance;
}

/// Build the Model Descriptor Card JSON the frontend uses to tokenize and
/// list the model. Paths point at the same files cpp_server itself loads so
/// the frontend tokenization matches exactly.
Json::Value buildMdcJson(const DiscoveryConfig& c) {
  Json::Value mdc(Json::objectValue);
  mdc["type"] = "Model";
  mdc["namespace"] = c.namespace_name;
  mdc["component"] = c.component;
  mdc["endpoint"] = c.endpoint;
  mdc["instance_id"] = static_cast<Json::UInt64>(c.instance_id);

  Json::Value card(Json::objectValue);
  card["display_name"] = c.model_name;
  card["slug"] = sanitizeSlug(c.model_name);
  card["source_path"] = c.model_path;

  const std::string configPath = c.model_path + "/config.json";
  const std::string tokenizerPath = c.model_path + "/tokenizer.json";
  const std::string tokenizerConfigPath =
      c.model_path + "/tokenizer_config.json";

  Json::Value modelInfo(Json::objectValue);
  Json::Value hfConfig(Json::objectValue);
  hfConfig["path"] = configPath;
  hfConfig["checksum"] = K_BLAKE3_PLACEHOLDER;
  modelInfo["hf_config_json"] = std::move(hfConfig);
  card["model_info"] = std::move(modelInfo);

  Json::Value tokenizer(Json::objectValue);
  Json::Value hfTok(Json::objectValue);
  hfTok["path"] = tokenizerPath;
  hfTok["checksum"] = K_BLAKE3_PLACEHOLDER;
  tokenizer["hf_tokenizer_json"] = std::move(hfTok);
  card["tokenizer"] = std::move(tokenizer);

  Json::Value promptFormatter(Json::objectValue);
  Json::Value hfTokCfg(Json::objectValue);
  hfTokCfg["path"] = tokenizerConfigPath;
  hfTokCfg["checksum"] = K_BLAKE3_PLACEHOLDER;
  promptFormatter["hf_tokenizer_config_json"] = std::move(hfTokCfg);
  card["prompt_formatter"] = std::move(promptFormatter);

  card["context_length"] = K_CONTEXT_LENGTH;
  card["kv_cache_block_size"] =
      static_cast<int>(tt::config::kvCacheBlockSize());
  card["migration_limit"] = 0;
  card["model_type"] = "Chat";
  card["model_input"] = "Tokens";

  Json::Value runtime(Json::objectValue);
  runtime["total_kv_blocks"] = Json::Value::null;
  runtime["max_num_seqs"] = Json::Value::null;
  runtime["max_num_batched_tokens"] = Json::Value::null;
  runtime["tool_call_parser"] = Json::Value::null;
  runtime["reasoning_parser"] = "deepseek_r1";
  runtime["exclude_tools_when_tool_choice_none"] = true;
  runtime["data_parallel_start_rank"] = 0;
  runtime["data_parallel_size"] = 1;
  runtime["enable_local_indexer"] = true;
  runtime["enable_eagle"] = false;
  card["runtime_config"] = std::move(runtime);

  card["media_decoder"] = Json::Value::null;
  card["media_fetcher"] = Json::Value::null;

  mdc["card_json"] = std::move(card);
  return mdc;
}

// ---------------------------------------------------------------------------
// Etcd backend
// ---------------------------------------------------------------------------

class EtcdDiscoveryRegistration : public DiscoveryRegistration {
 public:
  // NOTE: the parameter is named `config` (not `cfg`) on purpose so it does
  // not shadow the member `cfg`. With `DiscoveryConfig cfg` the second
  // initializer `client(... cfg.etcd_endpoints)` would read the
  // moved-from parameter and pass "" to EtcdClient.
  explicit EtcdDiscoveryRegistration(DiscoveryConfig config)
      : cfg(std::move(config)),
        client(std::make_unique<EtcdClient>(cfg.etcd_endpoints)) {}

  void registerSelf() override {
    if (leaseId == 0) {
      leaseId = client->leaseGrant(cfg.etcd_lease_ttl_secs);
      TT_LOG_INFO(
          "[DynamoDiscovery/etcd] Lease granted id={} ttl={}s endpoints={}",
          leaseId, cfg.etcd_lease_ttl_secs, cfg.etcd_endpoints);
    }
    publishKeys();
    TT_LOG_INFO(
        "[DynamoDiscovery/etcd] Registered key={} model={} endpoints={}",
        instanceKey(cfg), cfg.model_name, cfg.etcd_endpoints);
  }

  void keepAlive() override {
    if (leaseId == 0) {
      // Lost the lease (or never granted). Re-create from scratch.
      try {
        registerSelf();
      } catch (const std::exception& e) {
        TT_LOG_WARN("[DynamoDiscovery/etcd] re-register failed: {}", e.what());
      }
      return;
    }
    try {
      int64_t remaining = client->leaseKeepAlive(leaseId);
      if (remaining <= 0) {
        // etcd considered the lease expired; force a fresh register.
        TT_LOG_WARN(
            "[DynamoDiscovery/etcd] lease {} reported expired by etcd; "
            "re-registering",
            leaseId);
        leaseId = 0;
        registerSelf();
      }
    } catch (const std::exception& e) {
      // Likely transient (etcd hiccup, frontend restart). Drop the lease so
      // the next tick performs a clean re-register; do not crash.
      TT_LOG_WARN(
          "[DynamoDiscovery/etcd] keep-alive failed (lease {}): {} — will "
          "re-register",
          leaseId, e.what());
      leaseId = 0;
    }
  }

  void unregisterSelf() override {
    if (leaseId == 0) return;
    try {
      client->leaseRevoke(leaseId);
      TT_LOG_INFO("[DynamoDiscovery/etcd] Revoked lease {} (key={})", leaseId,
                  instanceKey(cfg));
    } catch (const std::exception& e) {
      TT_LOG_WARN("[DynamoDiscovery/etcd] revoke failed: {}", e.what());
    }
    leaseId = 0;
  }

  /// Keep-alive at half the lease TTL so a single missed tick doesn't expire
  /// the registration. Floor at 1s so misconfigured very-short TTLs don't
  /// busy-loop the keep-alive thread.
  int keepAliveIntervalSecs() const override {
    return std::max<int>(1, static_cast<int>(cfg.etcd_lease_ttl_secs / 2));
  }

 private:
  void publishKeys() {
    const std::string key = instanceKey(cfg);
    client->put("v1/instances/" + key, serializeJson(buildInstanceJson(cfg)),
                leaseId);
    client->put("v1/mdc/" + key, serializeJson(buildMdcJson(cfg)), leaseId);
  }

  DiscoveryConfig cfg;
  std::unique_ptr<EtcdClient> client;
  int64_t leaseId = 0;
};

}  // namespace

std::unique_ptr<DiscoveryRegistration> DiscoveryRegistration::create(
    const DiscoveryConfig& config) {
  return std::make_unique<EtcdDiscoveryRegistration>(config);
}

}  // namespace tt::dynamo
