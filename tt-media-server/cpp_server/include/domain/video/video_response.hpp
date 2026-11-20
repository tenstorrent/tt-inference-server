// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cmath>
#include <string>

#include "domain/base_response.hpp"

namespace tt::domain::video {

/**
 * Video result. `videoPath` is what the runner produced: a filesystem mp4 path
 * from a device runner, `mock://video/<task>` from the mock runner without
 * `TT_VIDEO_MOCK_WRITE_FILE`.
 */
struct VideoResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string videoPath;
  double generationTimeSeconds = 0.0;
  /** Submit -> result popped; for the task metrics (see VideoService). */
  double e2eSeconds = 0.0;
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
