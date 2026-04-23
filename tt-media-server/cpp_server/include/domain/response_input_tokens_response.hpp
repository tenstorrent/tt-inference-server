// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <string>

namespace tt::domain {

/**
 * Success response for POST /v1/responses/input_tokens.
 *
 * OpenAI shape: object "response.input_tokens" and integer input_tokens.
 * See
 * https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count.
 */
struct ResponseInputTokensResponse {
  static constexpr const char* kObjectType = "response.input_tokens";

  std::string object = kObjectType;
  int input_tokens = 0;

  Json::Value toJson() const {
    Json::Value j;
    j["object"] = object;
    j["input_tokens"] = input_tokens;
    return j;
  }
};

}  // namespace tt::domain
