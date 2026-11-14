// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstring>
#include <set>
#include <string>

namespace tt::domain::video {

/** Compact JSON in Python field order. JsonCpp stores object keys in a map, so
 *  `writeString` emits them alphabetically. Video routes instead follow
 *  `Job.to_public_dict` and `VideoGenerateRequest` field order. Keys this
 *  server adds on top of that schema are appended after the Python keys.
 */
inline std::string orderedJson(const Json::Value& value);

namespace {

inline std::string dumpAtom(const Json::Value& value) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  builder["commentStyle"] = "None";
  std::string text = Json::writeString(builder, value);
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  return text;
}

inline void appendField(std::string& out, bool& first, const std::string& key,
                        const std::string& encoded) {
  if (!first) out += ',';
  first = false;
  out += '"';
  out += key;
  out += "\":";
  out += encoded;
}

inline bool isRequestParameter(const std::string& key) {
  static const char* kKeys[] = {"prompt",
                                "negative_prompt",
                                "num_inference_steps",
                                "seed",
                                "aspect_ratio",
                                "height",
                                "width",
                                "fps",
                                "duration",
                                "num_frames"};
  for (const char* known : kKeys) {
    if (key == known) return true;
  }
  return false;
}

// Python model_dump order. Unset optionals are null. The shape validator
// fills a missing num_inference_steps with DEFAULT_VIDEO_INFERENCE_STEPS (20)
// before the job record is stored.
inline std::string writeRequestParameters(const Json::Value& params) {
  static const char* kOrder[] = {"prompt",
                                 "negative_prompt",
                                 "num_inference_steps",
                                 "seed",
                                 "aspect_ratio",
                                 "height",
                                 "width",
                                 "fps",
                                 "duration",
                                 "num_frames"};
  std::string out = "{";
  bool first = true;
  for (const char* key : kOrder) {
    Json::Value field = Json::nullValue;
    if (params.isObject() && params.isMember(key) && !params[key].isNull()) {
      field = params[key];
    } else if (std::strcmp(key, "num_inference_steps") == 0) {
      field = 20;
    }
    appendField(out, first, key, orderedJson(field));
  }
  if (params.isObject()) {
    for (const auto& name : params.getMemberNames()) {
      if (isRequestParameter(name)) continue;
      appendField(out, first, name, orderedJson(params[name]));
    }
  }
  out += '}';
  return out;
}

inline std::string writeObject(const Json::Value& obj,
                               std::initializer_list<const char*> order,
                               bool alwaysRequestParameters) {
  std::string out = "{";
  bool first = true;
  std::set<std::string> used;
  for (const char* key : order) {
    const bool present = obj.isMember(key);
    if (std::strcmp(key, "request_parameters") == 0 &&
        alwaysRequestParameters) {
      const Json::Value empty(Json::objectValue);
      appendField(out, first, key,
                  writeRequestParameters(present ? obj[key] : empty));
      used.insert(key);
      continue;
    }
    if (!present) continue;
    appendField(out, first, key, orderedJson(obj[key]));
    used.insert(key);
  }
  for (const auto& name : obj.getMemberNames()) {
    if (used.count(name) != 0) continue;
    appendField(out, first, name, orderedJson(obj[name]));
  }
  out += '}';
  return out;
}

}  // namespace

inline std::string orderedJson(const Json::Value& value) {
  if (value.isArray()) {
    std::string out = "[";
    bool first = true;
    for (const auto& item : value) {
      if (!first) out += ',';
      first = false;
      out += orderedJson(item);
    }
    out += ']';
    return out;
  }
  if (!value.isObject()) return dumpAtom(value);

  if (value.isMember("job_type")) {
    return writeObject(value,
                       {"id", "job_type", "status", "created_at", "model",
                        "request_parameters", "adapter_merge_job_ids", "org_id",
                        "completed_at", "error"},
                       true);
  }
  if (value.isMember("video_path")) {
    return writeObject(value, {"video_path", "generation_time"}, false);
  }
  if (value.isMember("detail")) {
    return writeObject(value, {"detail", "error"}, false);
  }
  if (value.isMember("deleted")) {
    return writeObject(value, {"id", "object", "deleted"}, false);
  }
  if (value.isMember("message") && value.isMember("type")) {
    return writeObject(value, {"message", "type", "param", "code"}, false);
  }
  if (value.isMember("code") && value.isMember("message")) {
    return writeObject(value, {"code", "message"}, false);
  }
  return writeObject(value, {}, false);
}

}  // namespace tt::domain::video
