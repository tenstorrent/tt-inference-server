// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "domain/base_request.hpp"
#include "domain/json_field.hpp"

namespace tt::domain {

// Same limits and default as Python config/constants.py.
constexpr int kMinVideoInferenceSteps = 4;
constexpr int kMaxVideoInferenceSteps = 50;
constexpr int kDefaultVideoInferenceSteps = 20;

/**
 * Text-to-video request. Field names match Python `VideoGenerateRequest`.
 * Shape and step limits that depend on the served model are not applied here;
 * this slice checks type, sign and the step range, and defaults the steps.
 */
struct VideoGenerateRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  std::string prompt;
  std::optional<std::string> negativePrompt;
  std::optional<int> numInferenceSteps;
  std::optional<int> seed;
  std::optional<std::string> aspectRatio;
  std::optional<int> height;
  std::optional<int> width;
  std::optional<float> fps;
  std::optional<float> duration;
  std::optional<int> numFrames;

  Json::Value toJson() const {
    Json::Value json;
    json["prompt"] = prompt;
    if (negativePrompt) json["negative_prompt"] = *negativePrompt;
    if (numInferenceSteps) json["num_inference_steps"] = *numInferenceSteps;
    if (seed) json["seed"] = *seed;
    if (aspectRatio) json["aspect_ratio"] = *aspectRatio;
    if (height) json["height"] = *height;
    if (width) json["width"] = *width;
    if (fps) json["fps"] = *fps;
    if (duration) json["duration"] = *duration;
    if (numFrames) json["num_frames"] = *numFrames;
    return json;
  }

  static VideoGenerateRequest fromJson(const Json::Value& json,
                                       uint32_t taskId) {
    auto present = [&](const char* key) {
      return json.isMember(key) && !json[key].isNull();
    };

    if (!present("prompt")) {
      throw std::invalid_argument("Missing required field: prompt");
    }

    VideoGenerateRequest req(taskId);
    req.prompt = json_field::getString(json["prompt"], "prompt");
    if (req.prompt.find_first_not_of(" \t\n\r") == std::string::npos) {
      throw std::invalid_argument("prompt must not be empty");
    }

    if (present("negative_prompt")) {
      req.negativePrompt =
          json_field::getString(json["negative_prompt"], "negative_prompt");
    }
    if (present("num_inference_steps")) {
      req.numInferenceSteps = json_field::getInt(json["num_inference_steps"],
                                                 "num_inference_steps");
      if (*req.numInferenceSteps < kMinVideoInferenceSteps ||
          *req.numInferenceSteps > kMaxVideoInferenceSteps) {
        throw std::invalid_argument("num_inference_steps must be between " +
                                    std::to_string(kMinVideoInferenceSteps) +
                                    " and " +
                                    std::to_string(kMaxVideoInferenceSteps));
      }
    } else {
      // Python resolves a missing value to the default, so the job echoes (and
      // the runner receives) what is actually generated.
      req.numInferenceSteps = kDefaultVideoInferenceSteps;
    }
    if (present("seed")) {
      req.seed = json_field::getInt(json["seed"], "seed");
    }
    if (present("aspect_ratio")) {
      req.aspectRatio =
          json_field::getString(json["aspect_ratio"], "aspect_ratio");
      if (req.aspectRatio->empty()) {
        throw std::invalid_argument("aspect_ratio must not be empty");
      }
    }
    if (present("height")) {
      req.height = json_field::getInt(json["height"], "height");
      if (*req.height <= 0) {
        throw std::invalid_argument("height must be positive");
      }
    }
    if (present("width")) {
      req.width = json_field::getInt(json["width"], "width");
      if (*req.width <= 0) {
        throw std::invalid_argument("width must be positive");
      }
    }
    if (present("fps")) {
      req.fps = json_field::getFloat(json["fps"], "fps");
      if (*req.fps <= 0.0F) {
        throw std::invalid_argument("fps must be positive");
      }
    }
    if (present("duration")) {
      req.duration = json_field::getFloat(json["duration"], "duration");
      if (*req.duration <= 0.0F) {
        throw std::invalid_argument("duration must be positive");
      }
    }
    if (present("num_frames")) {
      req.numFrames = json_field::getInt(json["num_frames"], "num_frames");
      if (*req.numFrames <= 0) {
        throw std::invalid_argument("num_frames must be positive");
      }
    }
    return req;
  }
};

}  // namespace tt::domain
