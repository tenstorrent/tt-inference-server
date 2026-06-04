// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>
#include <json/json.h>

#include <string>

namespace tt::api {

// OpenAI-compatible error: {"error": {"message", "type", "param", "code"}}
inline drogon::HttpResponsePtr errorResponse(
    drogon::HttpStatusCode status, const std::string& message,
    const std::string& type, const Json::Value& param = Json::nullValue,
    const Json::Value& code = Json::nullValue) {
  Json::Value body;
  body["error"]["message"] = message;
  body["error"]["type"] = type;
  body["error"]["param"] = param;
  body["error"]["code"] = code;
  auto resp = drogon::HttpResponse::newHttpJsonResponse(body);
  resp->setStatusCode(status);
  return resp;
}

// Stamps the end-to-end trace id onto a response as `X-Request-Id` so clients
// can correlate failures with server logs. No-op when traceId is empty.
// Returns the same response to allow inline wrapping at callback sites.
inline const drogon::HttpResponsePtr& withRequestId(
    const drogon::HttpResponsePtr& resp, const std::string& traceId) {
  if (!traceId.empty()) {
    resp->addHeader("X-Request-Id", traceId);
  }
  return resp;
}

}  // namespace tt::api
