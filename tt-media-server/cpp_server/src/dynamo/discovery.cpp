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

constexpr int kContextLength = 131072;
constexpr int kKvCacheBlockSize = 16;

/// URL-encode a path component (only `/` needs to be escaped for our keys).
std::string url_encode_path(const std::string& s) {
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
std::string blake3_placeholder() {
  return "blake3:0000000000000000000000000000000000000000000000000000000000"
         "000000";
}

/// Dynamo's slug validator rejects anything outside [a-z0-9_-]. HuggingFace
/// model ids have `/` and uppercase, so map them to `-` and lowercase.
std::string sanitize_slug(const std::string& s) {
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

std::string write_json(const Json::Value& v) {
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
  const std::string encoded_key = url_encode_path(key);

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

    const std::string filepath = dir + "/" + encoded_key;
    std::ofstream f(filepath);
    f << write_json(instance);
    f.close();

    if (!quiet) {
      TT_LOG_INFO("[DynamoDiscovery] Registered instance at {}", filepath);
    }
  }

  // ---- Model Descriptor Card (MDC) ----
  {
    const std::string dir = config.store_path + "/v1/mdc";
    fs::create_directories(dir);

    const std::string config_path = config.model_path + "/config.json";
    const std::string tokenizer_path = config.model_path + "/tokenizer.json";
    const std::string tokenizer_config_path =
        config.model_path + "/tokenizer_config.json";

    Json::Value mdc(Json::objectValue);
    mdc["type"] = "Model";
    mdc["namespace"] = config.namespace_name;
    mdc["component"] = config.component;
    mdc["endpoint"] = config.endpoint;
    mdc["instance_id"] = static_cast<Json::UInt64>(config.instance_id);

    Json::Value card(Json::objectValue);
    card["display_name"] = config.model_name;
    card["slug"] = sanitize_slug(config.model_name);
    card["source_path"] = config.model_path;

    Json::Value model_info(Json::objectValue);
    Json::Value hf_config(Json::objectValue);
    hf_config["path"] = config_path;
    hf_config["checksum"] = blake3_placeholder();
    model_info["hf_config_json"] = std::move(hf_config);
    card["model_info"] = std::move(model_info);

    Json::Value tokenizer(Json::objectValue);
    Json::Value hf_tok(Json::objectValue);
    hf_tok["path"] = tokenizer_path;
    hf_tok["checksum"] = blake3_placeholder();
    tokenizer["hf_tokenizer_json"] = std::move(hf_tok);
    card["tokenizer"] = std::move(tokenizer);

    Json::Value prompt_formatter(Json::objectValue);
    Json::Value hf_tok_cfg(Json::objectValue);
    hf_tok_cfg["path"] = tokenizer_config_path;
    hf_tok_cfg["checksum"] = blake3_placeholder();
    prompt_formatter["hf_tokenizer_config_json"] = std::move(hf_tok_cfg);
    card["prompt_formatter"] = std::move(prompt_formatter);

    card["context_length"] = kContextLength;
    card["kv_cache_block_size"] = kKvCacheBlockSize;
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

    const std::string filepath = dir + "/" + encoded_key;
    std::ofstream f(filepath);
    f << write_json(mdc);
    f.close();

    if (!quiet) {
      TT_LOG_INFO("[DynamoDiscovery] Registered MDC at {} (model={})", filepath,
                  config.model_name);
    }
  }
}

void unregister_discovery(const DiscoveryConfig& config) {
  const std::string key = config.namespace_name + "/" + config.component + "/" +
                          config.endpoint + "/" + config.instance_id_hex;
  const std::string encoded_key = url_encode_path(key);

  fs::remove(config.store_path + "/v1/instances/" + encoded_key);
  fs::remove(config.store_path + "/v1/mdc/" + encoded_key);
  TT_LOG_INFO("[DynamoDiscovery] Unregistered {}", key);
}

}  // namespace tt::dynamo
