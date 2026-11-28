// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
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
// Wan2.2 generates 81 frames; image_prompts[].frame_pos indexes into them.
constexpr int kVideoImagePromptFrames = 81;

/**
 * Video request. Field names match Python `VideoGenerateRequest`, plus
 * `image_prompts` (Python `VideoI2VGenerateRequest`, image-to-video and
 * MiniMax-H3 keyframes) and `references` (Python `VideoRef2VAGenerateRequest`).
 * Shape and step limits that depend on the served model are not applied here
 * (the Python runner refuses what its model cannot serve); this checks type,
 * sign, the step range and the media field shapes, and defaults the steps.
 */
struct VideoGenerateRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  std::string prompt;
  std::optional<std::string> negativePrompt;
  std::optional<int> numInferenceSteps;
  std::optional<int64_t> seed;
  std::optional<std::string> aspectRatio;
  std::optional<int> height;
  std::optional<int> width;
  std::optional<double> fps;
  std::optional<double> duration;
  std::optional<int> numFrames;
  // [{"image": "<base64>", "frame_pos": 0}, ...]
  std::optional<Json::Value> imagePrompts;
  // {"images": [{"b64": ...}], "videos": [...], "audios": [...]}
  std::optional<Json::Value> references;

  /** `withMedia` false replaces the base64 payloads by what they pin
   *  (frame_pos) or count, for the job record clients read back. */
  Json::Value toJson(bool withMedia = true) const {
    Json::Value json;
    json["prompt"] = prompt;
    if (negativePrompt) json["negative_prompt"] = *negativePrompt;
    if (numInferenceSteps) json["num_inference_steps"] = *numInferenceSteps;
    if (seed) json["seed"] = static_cast<Json::Int64>(*seed);
    if (aspectRatio) json["aspect_ratio"] = *aspectRatio;
    if (height) json["height"] = *height;
    if (width) json["width"] = *width;
    if (fps) json["fps"] = *fps;
    if (duration) json["duration"] = *duration;
    if (numFrames) json["num_frames"] = *numFrames;
    if (imagePrompts) {
      if (withMedia) {
        json["image_prompts"] = *imagePrompts;
      } else {
        Json::Value summary(Json::arrayValue);
        for (const auto& entry : *imagePrompts) {
          Json::Value item;
          item["frame_pos"] = entry.get("frame_pos", 0);
          summary.append(item);
        }
        json["image_prompts"] = summary;
      }
    }
    if (references) {
      if (withMedia) {
        json["references"] = *references;
      } else {
        Json::Value summary;
        for (const char* kind : {"images", "videos", "audios"}) {
          summary[kind] = (*references).get(kind, Json::arrayValue).size();
        }
        json["references"] = summary;
      }
    }
    return json;
  }

  static Json::Value parseImagePrompts(const Json::Value& value) {
    json_field::checkArray(value, "image_prompts");
    if (value.empty() || value.size() > kVideoImagePromptFrames) {
      throw std::invalid_argument("image_prompts must have 1 to " +
                                  std::to_string(kVideoImagePromptFrames) +
                                  " entries");
    }
    Json::Value out(Json::arrayValue);
    bool seen[kVideoImagePromptFrames + 1] = {};
    for (const auto& entry : value) {
      json_field::checkObject(entry, "image_prompts[]");
      if (!entry.isMember("image") ||
          json_field::getString(entry["image"], "image_prompts[].image")
              .empty()) {
        throw std::invalid_argument("image_prompts[].image must not be empty");
      }
      int framePos = 0;
      if (entry.isMember("frame_pos") && !entry["frame_pos"].isNull()) {
        framePos =
            json_field::getInt(entry["frame_pos"], "image_prompts[].frame_pos");
      }
      if (framePos < -1 || framePos >= kVideoImagePromptFrames) {
        throw std::invalid_argument(
            "image_prompts[].frame_pos must be in [-1, " +
            std::to_string(kVideoImagePromptFrames) + ")");
      }
      bool& slot = seen[framePos + 1];
      if (slot) {
        throw std::invalid_argument(
            "duplicate frame_pos=" + std::to_string(framePos) +
            " in image_prompts; each image must target a distinct frame");
      }
      slot = true;
      Json::Value item;
      item["image"] = entry["image"];
      item["frame_pos"] = framePos;
      out.append(item);
    }
    return out;
  }

  static Json::Value parseReferences(const Json::Value& value) {
    json_field::checkObject(value, "references");
    Json::Value out(Json::objectValue);
    size_t total = 0;
    for (const auto& key : value.getMemberNames()) {
      // Python ignores a group it does not know (an extra field of the model).
      if (key != "images" && key != "videos" && key != "audios") continue;
      const std::string field = "references." + key;
      json_field::checkArray(value[key], field.c_str());
      Json::Value list(Json::arrayValue);
      for (const auto& source : value[key]) {
        json_field::checkObject(source, field.c_str());
        const bool hasB64 = source.isMember("b64") && !source["b64"].isNull();
        const bool hasUrl = source.isMember("url") && !source["url"].isNull();
        // Python's MediaSource: exactly one of the two.
        if (hasB64 == hasUrl) {
          throw std::invalid_argument("provide exactly one of b64 or url");
        }
        if (hasUrl) {
          throw std::invalid_argument(
              field + "[].url is not supported by this server; send b64");
        }
        if (json_field::getString(source["b64"], "b64").empty()) {
          throw std::invalid_argument(field + "[].b64 must not be empty");
        }
        Json::Value item;
        item["b64"] = source["b64"];
        list.append(item);
      }
      total += list.size();
      out[key] = list;
    }
    if (total == 0) {
      // Python's check_reference_kinds
      throw std::invalid_argument(
          "ref2va needs at least one reference; use a t2va call for text-only "
          "requests");
    }
    return out;
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
      req.seed = json_field::getInt64(json["seed"], "seed");
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
      req.fps = json_field::getDouble(json["fps"], "fps");
      if (*req.fps <= 0.0) {
        throw std::invalid_argument("fps must be positive");
      }
    }
    if (present("duration")) {
      req.duration = json_field::getDouble(json["duration"], "duration");
      if (*req.duration <= 0.0) {
        throw std::invalid_argument("duration must be positive");
      }
    }
    if (present("num_frames")) {
      req.numFrames = json_field::getInt(json["num_frames"], "num_frames");
      if (*req.numFrames <= 0) {
        throw std::invalid_argument("num_frames must be positive");
      }
    }
    if (present("image_prompts")) {
      req.imagePrompts = parseImagePrompts(json["image_prompts"]);
    }
    if (present("references")) {
      req.references = parseReferences(json["references"]);
    }
    return req;
  }
};

}  // namespace tt::domain
