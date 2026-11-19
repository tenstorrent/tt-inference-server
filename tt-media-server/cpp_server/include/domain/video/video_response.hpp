// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cmath>
#include <string>

#include "domain/base_response.hpp"

namespace tt::domain::video {

/**
 * Synchronous video result. `videoPath` is a runner-produced reference
 * (`mock://video/<task>` in this slice, a filesystem mp4 path once a device
 * runner and the file response exist).
 */
struct VideoResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string videoPath;
  double generationTimeSeconds = 0.0;
  std::string error;

  Json::Value toOpenaiJson() const {
    Json::Value json;
    json["video_path"] = videoPath;
    if (generationTimeSeconds > 0.0) {
      json["generation_time"] =
          std::round(generationTimeSeconds * 100.0) / 100.0;
    }
    return json;
  }
};

}  // namespace tt::domain::video
