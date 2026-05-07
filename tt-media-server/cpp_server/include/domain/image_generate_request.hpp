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

  // Defaults below mirror Python's pydantic ImageGenerateRequest defaults
  // (domain/image_generate_request.py). They MUST stay in lockstep —
  // omitting them caused a real-world divergence where
  // `tt_sdxl.set_inference_params(...)` saw guidance_rescale=None instead
  // of 0.0, producing a different image with the same seed. They live on
  // the member declarations (rather than fromJson) so any other code path
  // that constructs an ImageGenerateRequest directly — e.g. the per-mode
  // warmupRequest() in sdxl_runner.cpp — picks them up automatically.
  std::optional<int> num_inference_steps = 20;
  std::optional<float> guidance_scale = 5.0F;
  std::optional<float> guidance_rescale = 0.0F;
  std::optional<int> seed;
  std::optional<int> number_of_images = 1;

  // SDXL crop conditioning: (top, left). Stored as int pair.
  std::optional<std::pair<int, int>> crop_coords_top_left =
      std::make_pair(0, 0);

  std::optional<std::vector<float>> timesteps;
  std::optional<std::vector<float>> sigmas;

  std::optional<std::string> lora_path;
  std::optional<float> lora_scale = 0.5F;

  // Output encoding hints.
  std::optional<std::string> image_return_format = "JPEG";  // "JPEG" or "PNG"
  std::optional<int> image_quality = 85;                    // 50..100

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
    // `present` is true only when the field is set AND not explicitly null.
    // Json::Value::asInt()/asFloat() silently coerce null -> 0 / 0.0, so
    // gating on isMember alone would let `{"seed": null}` overwrite the
    // pydantic default with 0. The json_field helpers below also reject
    // type mismatches with std::invalid_argument so the controller can
    // turn malformed payloads into 400s rather than 500s.
    auto present = [&](const char* key) {
      return json.isMember(key) && !json[key].isNull();
    };

    if (present("prompt"))
      req.prompt = json_field::getString(json["prompt"], "prompt");
    if (present("prompt_2"))
      req.prompt_2 = json_field::getString(json["prompt_2"], "prompt_2");
    if (present("negative_prompt"))
      req.negative_prompt =
          json_field::getString(json["negative_prompt"], "negative_prompt");
    if (present("negative_prompt_2"))
      req.negative_prompt_2 =
          json_field::getString(json["negative_prompt_2"], "negative_prompt_2");

    if (present("num_inference_steps"))
      req.num_inference_steps =
          json_field::getInt(json["num_inference_steps"], "num_inference_steps");
    if (present("guidance_scale"))
      req.guidance_scale =
          json_field::getFloat(json["guidance_scale"], "guidance_scale");
    if (present("guidance_rescale"))
      req.guidance_rescale =
          json_field::getFloat(json["guidance_rescale"], "guidance_rescale");
    if (present("seed"))
      req.seed = json_field::getInt(json["seed"], "seed");
    if (present("number_of_images"))
      req.number_of_images =
          json_field::getInt(json["number_of_images"], "number_of_images");

    if (present("crop_coords_top_left")) {
      const auto& arr = json["crop_coords_top_left"];
      json_field::checkArray(arr, "crop_coords_top_left");
      if (arr.size() != 2) {
        throw std::invalid_argument(
            "crop_coords_top_left must have exactly 2 elements");
      }
      req.crop_coords_top_left = std::make_pair(
          json_field::getInt(arr[0], "crop_coords_top_left[0]"),
          json_field::getInt(arr[1], "crop_coords_top_left[1]"));
    }

    auto readFloatArray = [](const Json::Value& arr, const char* field) {
      json_field::checkArray(arr, field);
      std::vector<float> out;
      out.reserve(arr.size());
      for (Json::ArrayIndex i = 0; i < arr.size(); ++i) {
        out.push_back(json_field::getFloat(arr[i], field));
      }
      return out;
    };
    if (present("timesteps"))
      req.timesteps = readFloatArray(json["timesteps"], "timesteps");
    if (present("sigmas"))
      req.sigmas = readFloatArray(json["sigmas"], "sigmas");

    if (present("lora_path"))
      req.lora_path = json_field::getString(json["lora_path"], "lora_path");
    if (present("lora_scale"))
      req.lora_scale = json_field::getFloat(json["lora_scale"], "lora_scale");

    if (present("image_return_format"))
      req.image_return_format = json_field::getString(
          json["image_return_format"], "image_return_format");
    if (present("image_quality"))
      req.image_quality =
          json_field::getInt(json["image_quality"], "image_quality");

    if (present("image"))
      req.image = json_field::getString(json["image"], "image");
    if (present("mask"))
      req.mask = json_field::getString(json["mask"], "mask");
    if (present("strength"))
      req.strength = json_field::getFloat(json["strength"], "strength");
    return req;
  }
};

}  // namespace tt::domain
