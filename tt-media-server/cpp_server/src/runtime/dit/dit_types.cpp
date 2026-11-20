// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/dit_types.hpp"

#include <set>

#include "runtime/dit/dit_env.hpp"

namespace tt::dit {

namespace {

bool present(const Json::Value& json, const char* key) {
  return json.isMember(key) && !json[key].isNull();
}

std::string str(const Json::Value& json, const char* key) {
  if (!json[key].isString()) {
    throw RequestError(std::string(key) + " must be a string");
  }
  return json[key].asString();
}

int integer(const Json::Value& json, const char* key) {
  if (!json[key].isIntegral()) {
    throw RequestError(std::string(key) + " must be an integer");
  }
  return json[key].asInt();
}

int64_t integer64(const Json::Value& json, const char* key) {
  if (!json[key].isIntegral() || !json[key].isInt64()) {
    throw RequestError(std::string(key) + " must be an integer");
  }
  return json[key].asInt64();
}

double number(const Json::Value& json, const char* key) {
  if (!json[key].isNumeric()) {
    throw RequestError(std::string(key) + " must be a number");
  }
  return json[key].asDouble();
}

}  // namespace

RunnerConfig RunnerConfig::fromEnv() {
  RunnerConfig cfg;
  cfg.modelRunner = envString("MODEL_RUNNER").value_or("");
  cfg.device = lowered(envString("DEVICE").value_or(""));
  cfg.modelWeightsPath = envString("MODEL_WEIGHTS_PATH").value_or("");
  cfg.outputDir = envString("TT_VIDEO_OUTPUT_DIR").value_or("/tmp/videos");
  return cfg;
}

DiTRequest DiTRequest::fromJson(const Json::Value& json) {
  if (!json.isObject()) throw RequestError("request must be an object");
  if (!present(json, "prompt") || trimmed(str(json, "prompt")).empty()) {
    throw RequestError("request has no prompt");
  }
  DiTRequest r;
  r.prompt = str(json, "prompt");
  if (present(json, "negative_prompt"))
    r.negativePrompt = str(json, "negative_prompt");
  if (present(json, "num_inference_steps"))
    r.numInferenceSteps = integer(json, "num_inference_steps");
  if (present(json, "seed")) r.seed = integer64(json, "seed");
  if (present(json, "fps")) r.fps = number(json, "fps");
  if (present(json, "height")) r.height = integer(json, "height");
  if (present(json, "width")) r.width = integer(json, "width");
  if (present(json, "num_frames")) r.numFrames = integer(json, "num_frames");
  if (present(json, "duration")) r.duration = number(json, "duration");
  if (present(json, "aspect_ratio")) r.aspectRatio = str(json, "aspect_ratio");
  if (present(json, "image")) r.image = str(json, "image");
  if (present(json, "task_id")) {
    r.taskId = json["task_id"].asString();
  }
  if (present(json, "image_prompts")) {
    if (!json["image_prompts"].isArray()) {
      throw RequestError("image_prompts must be an array");
    }
    for (const auto& entry : json["image_prompts"]) {
      if (!entry.isObject() || !present(entry, "image")) {
        throw RequestError("image_prompts[].image is required");
      }
      ImagePrompt p;
      p.image = str(entry, "image");
      if (present(entry, "frame_pos")) p.framePos = integer(entry, "frame_pos");
      r.imagePrompts.push_back(std::move(p));
    }
  }
  if (present(json, "references")) {
    if (!json["references"].isObject()) {
      throw RequestError("references must be an object");
    }
    r.references = json["references"];
  }
  return r;
}

Json::Value DiTRequest::toJson() const {
  Json::Value j;
  j["prompt"] = prompt;
  if (negativePrompt) j["negative_prompt"] = *negativePrompt;
  if (numInferenceSteps) j["num_inference_steps"] = *numInferenceSteps;
  if (seed) j["seed"] = static_cast<Json::Int64>(*seed);
  if (fps) j["fps"] = *fps;
  if (height) j["height"] = *height;
  if (width) j["width"] = *width;
  if (numFrames) j["num_frames"] = *numFrames;
  if (duration) j["duration"] = *duration;
  if (aspectRatio) j["aspect_ratio"] = *aspectRatio;
  if (image) j["image"] = *image;
  if (!taskId.empty()) j["task_id"] = taskId;
  if (!imagePrompts.empty()) {
    Json::Value list(Json::arrayValue);
    for (const auto& p : imagePrompts) {
      Json::Value item;
      item["image"] = p.image;
      item["frame_pos"] = p.framePos;
      list.append(item);
    }
    j["image_prompts"] = list;
  }
  if (references.isObject()) j["references"] = references;
  return j;
}

void validateImagePrompts(const std::vector<ImagePrompt>& prompts,
                          int numFrames) {
  if (prompts.empty()) {
    throw RequestError("this model needs image_prompts (at least one image)");
  }
  if (static_cast<int>(prompts.size()) > numFrames) {
    throw RequestError("at most " + std::to_string(numFrames) +
                       " image_prompts are accepted");
  }
  std::set<int> seen;
  for (const auto& p : prompts) {
    if (p.image.empty()) {
      throw RequestError("image_prompts[].image must not be empty");
    }
    if (p.framePos < -1 || p.framePos >= numFrames) {
      throw RequestError("image_prompts[].frame_pos must be in [-1, " +
                         std::to_string(numFrames) + "); got " +
                         std::to_string(p.framePos));
    }
    if (!seen.insert(p.framePos).second) {
      throw RequestError("duplicate frame_pos=" + std::to_string(p.framePos) +
                         " in image_prompts; each image must target a "
                         "distinct frame");
    }
  }
}

}  // namespace tt::dit
