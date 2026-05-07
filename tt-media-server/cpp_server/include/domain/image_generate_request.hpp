// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/json_field.hpp"

namespace tt::domain {

/**
 * OpenAI-compatible image generation request. Mirrors the Python
 * ImageGenerateRequest in `domain/image_generate_request.py` and is also the
 * superset shape consumed by the SDXL family of runners.
 *
 * Per-modality validation (e.g. min num_inference_steps for Flux) lives
 * downstream where the active model_runner is known.
 */
struct ImageGenerateRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  std::string prompt;
  std::optional<std::string> prompt_2;
  std::optional<std::string> negative_prompt;
  std::optional<std::string> negative_prompt_2;

  std::optional<int> num_inference_steps;
  std::optional<float> guidance_scale;
  std::optional<float> guidance_rescale;
  std::optional<int> seed;
  std::optional<int> number_of_images;

  // SDXL crop conditioning: (top, left). Stored as int pair.
  std::optional<std::pair<int, int>> crop_coords_top_left;

  std::optional<std::vector<float>> timesteps;
  std::optional<std::vector<float>> sigmas;

  std::optional<std::string> lora_path;
  std::optional<float> lora_scale;

  // Output encoding hints.
  std::optional<std::string> image_return_format;  // "JPEG" or "PNG"
  std::optional<int> image_quality;                // 50..100

  // Image-to-image / edit fields. Mirror the Python ImageToImageRequest
  // and ImageEditRequest schemas; carried on the base struct so the runner
  // base class can stay agnostic to the exact request type. The controller
  // is responsible for enforcing presence of `image` for img2img and
  // `image` + `mask` for edit endpoints.
  std::optional<std::string> image;  // base64-encoded input image
  std::optional<std::string> mask;   // base64-encoded mask (edit/inpaint)
  std::optional<float> strength;

  static ImageGenerateRequest fromJson(const Json::Value& json,
                                       uint32_t taskId) {
    ImageGenerateRequest req(taskId);
    // Defaults below mirror Python's pydantic ImageGenerateRequest defaults
    // (domain/image_generate_request.py). They MUST stay in lockstep —
    // omitting them caused a real-world divergence where
    // `tt_sdxl.set_inference_params(...)` saw guidance_rescale=None instead
    // of 0.0, producing a different image with the same seed.
    req.num_inference_steps = 20;
    req.guidance_scale = 5.0F;
    req.guidance_rescale = 0.0F;
    req.number_of_images = 1;
    req.image_return_format = "JPEG";
    req.image_quality = 85;
    req.lora_scale = 0.5F;
    req.crop_coords_top_left = std::make_pair(0, 0);

    if (json.isMember("prompt"))
      req.prompt = json_field::getString(json["prompt"], "prompt");
    if (json.isMember("prompt_2"))
      req.prompt_2 = json_field::getString(json["prompt_2"], "prompt_2");
    if (json.isMember("negative_prompt"))
      req.negative_prompt =
          json_field::getString(json["negative_prompt"], "negative_prompt");
    if (json.isMember("negative_prompt_2"))
      req.negative_prompt_2 =
          json_field::getString(json["negative_prompt_2"], "negative_prompt_2");

    if (json.isMember("num_inference_steps"))
      req.num_inference_steps = json["num_inference_steps"].asInt();
    if (json.isMember("guidance_scale"))
      req.guidance_scale = json["guidance_scale"].asFloat();
    if (json.isMember("guidance_rescale"))
      req.guidance_rescale = json["guidance_rescale"].asFloat();
    if (json.isMember("seed")) req.seed = json["seed"].asInt();
    if (json.isMember("number_of_images"))
      req.number_of_images = json["number_of_images"].asInt();

    if (json.isMember("crop_coords_top_left") &&
        json["crop_coords_top_left"].isArray() &&
        json["crop_coords_top_left"].size() == 2) {
      req.crop_coords_top_left =
          std::make_pair(json["crop_coords_top_left"][0].asInt(),
                         json["crop_coords_top_left"][1].asInt());
    }

    auto readFloatArray = [](const Json::Value& arr) {
      std::vector<float> out;
      out.reserve(arr.size());
      for (Json::ArrayIndex i = 0; i < arr.size(); ++i) {
        out.push_back(arr[i].asFloat());
      }
      return out;
    };
    if (json.isMember("timesteps") && json["timesteps"].isArray())
      req.timesteps = readFloatArray(json["timesteps"]);
    if (json.isMember("sigmas") && json["sigmas"].isArray())
      req.sigmas = readFloatArray(json["sigmas"]);

    if (json.isMember("lora_path"))
      req.lora_path = json_field::getString(json["lora_path"], "lora_path");
    if (json.isMember("lora_scale"))
      req.lora_scale = json["lora_scale"].asFloat();

    if (json.isMember("image_return_format"))
      req.image_return_format = json_field::getString(
          json["image_return_format"], "image_return_format");
    if (json.isMember("image_quality"))
      req.image_quality = json["image_quality"].asInt();

    if (json.isMember("image"))
      req.image = json_field::getString(json["image"], "image");
    if (json.isMember("mask"))
      req.mask = json_field::getString(json["mask"], "mask");
    if (json.isMember("strength"))
      req.strength = json["strength"].asFloat();
    return req;
  }
};

}  // namespace tt::domain
