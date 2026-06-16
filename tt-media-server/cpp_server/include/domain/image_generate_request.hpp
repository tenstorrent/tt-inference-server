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
 * OpenAI-compatible image generation request. Carries the superset of fields
 * for text-to-image, image-to-image, and edit/inpaint endpoints; the
 * controller enforces the presence of mode-specific fields.
 */
struct ImageGenerateRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  std::string prompt;
  std::optional<std::string> prompt_2;
  std::optional<std::string> negative_prompt;
  std::optional<std::string> negative_prompt_2;

  std::optional<int> num_inference_steps = 20;
  std::optional<float> guidance_scale = 5.0F;
  std::optional<float> guidance_rescale = 0.0F;
  std::optional<int> seed;
  std::optional<int> number_of_images = 1;

  std::optional<std::pair<int, int>> crop_coords_top_left =
      std::make_pair(0, 0);

  std::optional<std::vector<float>> timesteps;
  std::optional<std::vector<float>> sigmas;

  std::optional<std::string> lora_path;
  std::optional<float> lora_scale = 0.5F;

  std::optional<std::string> image_return_format = "JPEG";  // "JPEG" or "PNG"
  std::optional<int> image_quality = 85;

  std::optional<std::string> image;
  std::optional<std::string> mask;
  std::optional<float> strength;

  Json::Value toJson() const {
    Json::Value json;
    json["prompt"] = prompt;
    if (prompt_2) json["prompt_2"] = *prompt_2;
    if (negative_prompt) json["negative_prompt"] = *negative_prompt;
    if (negative_prompt_2) json["negative_prompt_2"] = *negative_prompt_2;
    if (num_inference_steps) json["num_inference_steps"] = *num_inference_steps;
    if (guidance_scale) json["guidance_scale"] = *guidance_scale;
    if (guidance_rescale) json["guidance_rescale"] = *guidance_rescale;
    if (seed) json["seed"] = *seed;
    if (number_of_images) json["number_of_images"] = *number_of_images;
    if (crop_coords_top_left) {
      Json::Value coords(Json::arrayValue);
      coords.append(crop_coords_top_left->first);
      coords.append(crop_coords_top_left->second);
      json["crop_coords_top_left"] = std::move(coords);
    }
    if (timesteps) {
      Json::Value values(Json::arrayValue);
      for (float value : *timesteps) values.append(value);
      json["timesteps"] = std::move(values);
    }
    if (sigmas) {
      Json::Value values(Json::arrayValue);
      for (float value : *sigmas) values.append(value);
      json["sigmas"] = std::move(values);
    }
    if (lora_path) json["lora_path"] = *lora_path;
    if (lora_scale) json["lora_scale"] = *lora_scale;
    if (image_return_format) json["image_return_format"] = *image_return_format;
    if (image_quality) json["image_quality"] = *image_quality;
    if (image) json["image"] = *image;
    if (mask) json["mask"] = *mask;
    if (strength) json["strength"] = *strength;
    return json;
  }

  static ImageGenerateRequest fromJson(const Json::Value& json,
                                       uint32_t taskId) {
    ImageGenerateRequest req(taskId);
    // Treat explicit nulls as absent so they don't clobber pydantic defaults
    // via Json::Value::asInt()/asFloat() silent-zero coercion.
    auto present = [&](const char* key) {
      return json.isMember(key) && !json[key].isNull();
    };

    if (!present("prompt")) {
      throw std::invalid_argument("Missing required field: prompt");
    }
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
      req.num_inference_steps = json_field::getInt(json["num_inference_steps"],
                                                   "num_inference_steps");
    if (present("guidance_scale"))
      req.guidance_scale =
          json_field::getFloat(json["guidance_scale"], "guidance_scale");
    if (present("guidance_rescale"))
      req.guidance_rescale =
          json_field::getFloat(json["guidance_rescale"], "guidance_rescale");
    if (present("seed")) req.seed = json_field::getInt(json["seed"], "seed");
    if (present("number_of_images")) {
      req.number_of_images =
          json_field::getInt(json["number_of_images"], "number_of_images");
      // The C++ runner produces one image per request. The Python service
      // emulates number_of_images > 1 via repeated runs with incremented
      // seeds; that loop is not ported here, so reject up front instead of
      // silently dropping all but the first image.
      if (req.number_of_images.value() != 1) {
        throw std::invalid_argument(
            "number_of_images must be 1 (multi-image batching is not "
            "supported by the C++ runner)");
      }
    }

    if (present("crop_coords_top_left")) {
      const auto& arr = json["crop_coords_top_left"];
      json_field::checkArray(arr, "crop_coords_top_left");
      if (arr.size() != 2) {
        throw std::invalid_argument(
            "crop_coords_top_left must have exactly 2 elements");
      }
      req.crop_coords_top_left =
          std::make_pair(json_field::getInt(arr[0], "crop_coords_top_left[0]"),
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

    if (present("image_return_format")) {
      const auto format = json_field::getString(json["image_return_format"],
                                                "image_return_format");
      if (format != "JPEG" && format != "PNG") {
        throw std::invalid_argument(
            "image_return_format must be either 'JPEG' or 'PNG'");
      }
      req.image_return_format = format;
    }
    if (present("image_quality")) {
      const int quality =
          json_field::getInt(json["image_quality"], "image_quality");
      if (quality < 50 || quality > 100) {
        throw std::invalid_argument(
            "image_quality must be between 50 and 100 (inclusive)");
      }
      req.image_quality = quality;
    }

    if (present("image"))
      req.image = json_field::getString(json["image"], "image");
    if (present("mask")) req.mask = json_field::getString(json["mask"], "mask");
    if (present("strength"))
      req.strength = json_field::getFloat(json["strength"], "strength");
    return req;
  }
};

}  // namespace tt::domain
