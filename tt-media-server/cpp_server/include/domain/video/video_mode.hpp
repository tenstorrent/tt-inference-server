// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cctype>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace tt::domain::video {

/** Python `use_async_video`. True: POST returns 202 and a job id. False: POST
 *  stays open and returns the mp4 path when generation finishes. */
inline std::optional<bool> parseAsyncFlag(const std::string& text) {
  std::string folded;
  folded.reserve(text.size());
  for (unsigned char ch : text) {
    if (ch == ' ' || ch == '\t') continue;
    folded.push_back(static_cast<char>(std::tolower(ch)));
  }
  if (folded == "1" || folded == "true" || folded == "yes" || folded == "on") {
    return true;
  }
  if (folded == "0" || folded == "false" || folded == "no" || folded == "off") {
    return false;
  }
  return std::nullopt;
}

/** `USE_ASYNC_VIDEO`, same name as the Python setting. Unset means async. */
inline bool asyncVideoFromEnv() {
  const char* value = std::getenv("USE_ASYNC_VIDEO");
  if (value == nullptr || *value == '\0') return true;
  const std::optional<bool> parsed = parseAsyncFlag(value);
  return parsed.value_or(true);
}

/**
 * Both modes stay available on one server. A query `async` or a JSON `async`
 * field on this request wins. Otherwise the env default applies.
 * Throws std::invalid_argument for an unrecognised query value or JSON value
 * (the controller answers 422) instead of silently falling through.
 */
inline bool videoRequestIsAsync(const std::string& queryAsync,
                                const Json::Value& body) {
  if (!queryAsync.empty()) {
    if (const std::optional<bool> parsed = parseAsyncFlag(queryAsync)) {
      return *parsed;
    }
    throw std::invalid_argument(
        "async query parameter must be one of true/false/1/0/yes/no/on/off");
  }
  if (body.isMember("async") && !body["async"].isNull()) {
    const Json::Value& field = body["async"];
    if (field.isBool()) return field.asBool();
    if (field.isString()) {
      if (const std::optional<bool> parsed = parseAsyncFlag(field.asString())) {
        return *parsed;
      }
    }
    throw std::invalid_argument(
        "async must be a boolean or one of true/false/1/0/yes/no/on/off");
  }
  return asyncVideoFromEnv();
}

}  // namespace tt::domain::video
