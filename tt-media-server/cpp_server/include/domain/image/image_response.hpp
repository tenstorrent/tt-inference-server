// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "domain/base_response.hpp"

namespace tt::domain::image {

/**
 * Response for image generation / image-to-image / edit endpoints.
 * Carries the produced images as a list of base64-encoded strings, matching
 * the shape returned by the Python `open_ai_api/image.py` handlers.
 */
struct ImageResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::vector<std::string> images;
  double generation_time_seconds = 0.0;
  std::string error;

  Json::Value toOpenaiJson() const {
    Json::Value json;
    Json::Value imagesArr(Json::arrayValue);
    for (const auto& img : images) imagesArr.append(img);
    json["images"] = std::move(imagesArr);
    if (generation_time_seconds > 0.0) {
      json["generation_time"] =
          std::round(generation_time_seconds * 100.0) / 100.0;
    }
    return json;
  }
};

}  // namespace tt::domain::image
