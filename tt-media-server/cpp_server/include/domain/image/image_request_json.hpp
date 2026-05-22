// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include "domain/image_generate_request.hpp"

namespace tt::domain::image {

inline Json::Value toJson(const tt::domain::ImageGenerateRequest& request) {
  Json::Value json;
  json["prompt"] = request.prompt;
  if (request.prompt_2) json["prompt_2"] = *request.prompt_2;
  if (request.negative_prompt) json["negative_prompt"] = *request.negative_prompt;
  if (request.negative_prompt_2) {
    json["negative_prompt_2"] = *request.negative_prompt_2;
  }
  if (request.num_inference_steps) {
    json["num_inference_steps"] = *request.num_inference_steps;
  }
  if (request.guidance_scale) json["guidance_scale"] = *request.guidance_scale;
  if (request.guidance_rescale) {
    json["guidance_rescale"] = *request.guidance_rescale;
  }
  if (request.seed) json["seed"] = *request.seed;
  if (request.number_of_images) json["number_of_images"] = *request.number_of_images;
  if (request.crop_coords_top_left) {
    Json::Value coords(Json::arrayValue);
    coords.append(request.crop_coords_top_left->first);
    coords.append(request.crop_coords_top_left->second);
    json["crop_coords_top_left"] = std::move(coords);
  }
  if (request.timesteps) {
    Json::Value values(Json::arrayValue);
    for (float value : *request.timesteps) values.append(value);
    json["timesteps"] = std::move(values);
  }
  if (request.sigmas) {
    Json::Value values(Json::arrayValue);
    for (float value : *request.sigmas) values.append(value);
    json["sigmas"] = std::move(values);
  }
  if (request.lora_path) json["lora_path"] = *request.lora_path;
  if (request.lora_scale) json["lora_scale"] = *request.lora_scale;
  if (request.image_return_format) {
    json["image_return_format"] = *request.image_return_format;
  }
  if (request.image_quality) json["image_quality"] = *request.image_quality;
  if (request.image) json["image"] = *request.image;
  if (request.mask) json["mask"] = *request.mask;
  if (request.strength) json["strength"] = *request.strength;
  return json;
}

}  // namespace tt::domain::image
