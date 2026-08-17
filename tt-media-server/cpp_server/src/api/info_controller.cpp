// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/info_controller.hpp"

#include <json/json.h>

#include <string>

#include "config/build_info.hpp"

namespace tt::api {

void InfoController::info(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  Json::Value response;
  response["tt_inference_server"]["version"] =
      std::string{tt::config::kInferenceServerVersion};
  response["tt_inference_server"]["commit"] =
      std::string{tt::config::kInferenceServerCommit};
  response["tt_blaze"]["commit"] = std::string{tt::config::kTtBlazeCommit};
  response["tt_metal"]["commit"] = std::string{tt::config::kTtMetalCommit};

  callback(drogon::HttpResponse::newHttpJsonResponse(response));
}

}  // namespace tt::api
