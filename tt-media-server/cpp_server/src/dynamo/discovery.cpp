// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/discovery.hpp"

#include <json/json.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace fs = std::filesystem;

namespace {

constexpr int K_CONTEXT_LENGTH = 131072;
constexpr int K_KV_CACHE_BLOCK_SIZE = 16;

/// URL-encode a path component (only `/` needs to be escaped for our keys).
std::string urlEncodePath(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '/') {
      out += "%2F";
    } else {
      out += c;
    }
  }
  return out;
}

/// Frontend reads the checksum but doesn't validate it for routing — it's
/// used only for cache invalidation.
std::string blake3Placeholder() {
  return "blake3:0000000000000000000000000000000000000000000000000000000000"
         "000000";
}

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

std::string writeJson(const Json::Value& v) {
  Json::StreamWriterBuilder b;
  b["indentation"] = "";
  b["emitUTF8"] = true;
  return Json::writeString(b, v);
}

}  // namespace

void register_discovery(const DiscoveryConfig& config, bool quiet) {
  // Key = "namespace/component/endpoint/instance_id_hex" (URL-encoded).
  const std::string key = config.namespace_name + "/" + config.component + "/" +
                          config.endpoint + "/" + config.instance_id_hex;
  const std::string encodedKey = urlEncodePath(key);

  // ---- instance JSON ----
  {
    const std::string dir = config.store_path + "/v1/instances";
    fs::create_directories(dir);

    Json::Value instance(Json::objectValue);
    instance["type"] = "Endpoint";
    instance["component"] = config.component;
    instance["endpoint"] = config.endpoint;
    instance["namespace"] = config.namespace_name;
    instance["instance_id"] = static_cast<Json::UInt64>(config.instance_id);

    Json::Value transport(Json::objectValue);
    transport["tcp"] = config.tcp_address;
    instance["transport"] = std::move(transport);
    instance["device_type"] = "cuda";

    const std::string filePath = dir + "/" + encodedKey;
    std::ofstream f(filePath);
    f << writeJson(instance);
    f.close();

    if (!quiet) {
      TT_LOG_INFO("[DynamoDiscovery] Registered instance at {}", filePath);
    }
  }

  // ---- Model Descriptor Card (MDC) ----
  {
    const std::string dir = config.store_path + "/v1/mdc";
    fs::create_directories(dir);

    const std::string configPath = config.model_path + "/config.json";
    const std::string tokenizerPath = config.model_path + "/tokenizer.json";
    const std::string tokenizerConfigPath =
        config.model_path + "/tokenizer_config.json";

    Json::Value mdc(Json::objectValue);
    mdc["type"] = "Model";
    mdc["namespace"] = config.namespace_name;
    mdc["component"] = config.component;
    mdc["endpoint"] = config.endpoint;
    mdc["instance_id"] = static_cast<Json::UInt64>(config.instance_id);

    Json::Value card(Json::objectValue);
    card["display_name"] = config.model_name;
    card["slug"] = sanitizeSlug(config.model_name);
    card["source_path"] = config.model_path;

    Json::Value modelInfo(Json::objectValue);
    Json::Value hfConfig(Json::objectValue);
    hfConfig["path"] = configPath;
    hfConfig["checksum"] = blake3Placeholder();
    modelInfo["hf_config_json"] = std::move(hfConfig);
    card["model_info"] = std::move(modelInfo);

    Json::Value tokenizer(Json::objectValue);
    Json::Value hfTok(Json::objectValue);
    hfTok["path"] = tokenizerPath;
    hfTok["checksum"] = blake3Placeholder();
    tokenizer["hf_tokenizer_json"] = std::move(hfTok);
    card["tokenizer"] = std::move(tokenizer);

    Json::Value promptFormatter(Json::objectValue);
    Json::Value hfTokCfg(Json::objectValue);
    hfTokCfg["path"] = tokenizerConfigPath;
    hfTokCfg["checksum"] = blake3Placeholder();
    promptFormatter["hf_tokenizer_config_json"] = std::move(hfTokCfg);
    card["prompt_formatter"] = std::move(promptFormatter);

    card["context_length"] = K_CONTEXT_LENGTH;
    card["kv_cache_block_size"] = K_KV_CACHE_BLOCK_SIZE;
    card["migration_limit"] = 0;
    card["model_type"] = "Chat";
    card["model_input"] = "Tokens";

    Json::Value runtime(Json::objectValue);
    runtime["total_kv_blocks"] = Json::Value::null;
    runtime["max_num_seqs"] = Json::Value::null;
    runtime["max_num_batched_tokens"] = Json::Value::null;
    runtime["tool_call_parser"] = Json::Value::null;
    runtime["reasoning_parser"] = Json::Value::null;
    runtime["exclude_tools_when_tool_choice_none"] = true;
    runtime["data_parallel_start_rank"] = 0;
    runtime["data_parallel_size"] = 1;
    runtime["enable_local_indexer"] = true;
    runtime["enable_eagle"] = false;
    card["runtime_config"] = std::move(runtime);

    card["media_decoder"] = Json::Value::null;
    card["media_fetcher"] = Json::Value::null;

    mdc["card_json"] = std::move(card);

    const std::string filePath = dir + "/" + encodedKey;
    std::ofstream f(filePath);
    f << writeJson(mdc);
    f.close();

    if (!quiet) {
      TT_LOG_INFO("[DynamoDiscovery] Registered MDC at {} (model={})", filePath,
                  config.model_name);
    }
  }
}

void unregister_discovery(const DiscoveryConfig& config) {
  const std::string key = config.namespace_name + "/" + config.component + "/" +
                          config.endpoint + "/" + config.instance_id_hex;
  const std::string encodedKey = urlEncodePath(key);

  fs::remove(config.store_path + "/v1/instances/" + encodedKey);
  fs::remove(config.store_path + "/v1/mdc/" + encodedKey);
  TT_LOG_INFO("[DynamoDiscovery] Unregistered {}", key);
}

}  // namespace tt::dynamo
