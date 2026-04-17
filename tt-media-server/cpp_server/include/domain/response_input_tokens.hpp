// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <string>

#include "domain/responses_request.hpp"

namespace tt::domain {

/**
 * Request body for POST /v1/responses/input_tokens.
 * Matches OpenAI "Get input token counts" (responses input_tokens count): same
 * fields as creating a response (conversation, input, instructions, model,
 * tools, etc.); see Responses API reference.
 */
struct ResponseInputTokensRequest {
  ResponsesRequest responses;

  static ResponseInputTokensRequest fromJson(const Json::Value& json,
                                             uint32_t taskId) {
    return ResponseInputTokensRequest{ResponsesRequest::fromJson(json, taskId)};
  }
};

/**
 * Success response for input token counting.
 * OpenAI shape: object "response.input_tokens" and integer input_tokens.
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
