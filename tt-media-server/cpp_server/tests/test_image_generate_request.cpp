// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>
#include <json/json.h>

#include <stdexcept>
#include <string>

#include "domain/image_generate_request.hpp"

using tt::domain::ImageGenerateRequest;

namespace {

constexpr uint32_t TEST_TASK_ID = 42;

Json::Value minimalGenerateJson(const std::string& prompt = "a cat") {
  Json::Value j;
  j["prompt"] = prompt;
  return j;
}

}  // namespace

TEST(ImageGenerateRequestParseTest, MinimalJsonAppliesDefaults) {
  auto req =
      ImageGenerateRequest::fromJson(minimalGenerateJson(), TEST_TASK_ID);

  EXPECT_EQ(req.task_id, TEST_TASK_ID);
  EXPECT_EQ(req.prompt, "a cat");

  // Pydantic-aligned defaults must survive a minimal payload.
  ASSERT_TRUE(req.num_inference_steps.has_value());
  EXPECT_EQ(*req.num_inference_steps, 20);
  ASSERT_TRUE(req.guidance_scale.has_value());
  EXPECT_FLOAT_EQ(*req.guidance_scale, 5.0F);
  ASSERT_TRUE(req.guidance_rescale.has_value());
  EXPECT_FLOAT_EQ(*req.guidance_rescale, 0.0F);
  ASSERT_TRUE(req.number_of_images.has_value());
  EXPECT_EQ(*req.number_of_images, 1);
  ASSERT_TRUE(req.crop_coords_top_left.has_value());
  EXPECT_EQ(req.crop_coords_top_left->first, 0);
  EXPECT_EQ(req.crop_coords_top_left->second, 0);
  ASSERT_TRUE(req.image_return_format.has_value());
  EXPECT_EQ(*req.image_return_format, "JPEG");
  ASSERT_TRUE(req.image_quality.has_value());
  EXPECT_EQ(*req.image_quality, 85);
  ASSERT_TRUE(req.lora_scale.has_value());
  EXPECT_FLOAT_EQ(*req.lora_scale, 0.5F);

  EXPECT_FALSE(req.seed.has_value());
  EXPECT_FALSE(req.negative_prompt.has_value());
  EXPECT_FALSE(req.image.has_value());
  EXPECT_FALSE(req.mask.has_value());
  EXPECT_FALSE(req.strength.has_value());
}

TEST(ImageGenerateRequestParseTest, ExplicitNullPreservesDefault) {
  // Json::Value's silent zero-coercion (asInt() on null returns 0) would
  // clobber the pydantic defaults; fromJson must treat null as absent.
  auto json = minimalGenerateJson();
  json["num_inference_steps"] = Json::Value::null;
  json["guidance_scale"] = Json::Value::null;
  json["seed"] = Json::Value::null;

  auto req = ImageGenerateRequest::fromJson(json, TEST_TASK_ID);

  ASSERT_TRUE(req.num_inference_steps.has_value());
  EXPECT_EQ(*req.num_inference_steps, 20);
  ASSERT_TRUE(req.guidance_scale.has_value());
  EXPECT_FLOAT_EQ(*req.guidance_scale, 5.0F);
  EXPECT_FALSE(req.seed.has_value());
}

TEST(ImageGenerateRequestParseTest, ExplicitFieldsOverrideDefaults) {
  auto json = minimalGenerateJson("a hedgehog");
  json["prompt_2"] = "secondary";
  json["negative_prompt"] = "blurry";
  json["negative_prompt_2"] = "low quality";
  json["num_inference_steps"] = 30;
  json["guidance_scale"] = 7.5;
  json["guidance_rescale"] = 0.2;
  json["seed"] = 1234;
  json["number_of_images"] = 2;
  json["lora_path"] = "/tmp/lora.safetensors";
  json["lora_scale"] = 0.75;
  json["image_return_format"] = "PNG";
  json["image_quality"] = 95;

  auto req = ImageGenerateRequest::fromJson(json, TEST_TASK_ID);

  EXPECT_EQ(req.prompt, "a hedgehog");
  ASSERT_TRUE(req.prompt_2.has_value());
  EXPECT_EQ(*req.prompt_2, "secondary");
  ASSERT_TRUE(req.negative_prompt.has_value());
  EXPECT_EQ(*req.negative_prompt, "blurry");
  ASSERT_TRUE(req.negative_prompt_2.has_value());
  EXPECT_EQ(*req.negative_prompt_2, "low quality");
  EXPECT_EQ(*req.num_inference_steps, 30);
  EXPECT_FLOAT_EQ(*req.guidance_scale, 7.5F);
  EXPECT_FLOAT_EQ(*req.guidance_rescale, 0.2F);
  ASSERT_TRUE(req.seed.has_value());
  EXPECT_EQ(*req.seed, 1234);
  EXPECT_EQ(*req.number_of_images, 2);
  ASSERT_TRUE(req.lora_path.has_value());
  EXPECT_EQ(*req.lora_path, "/tmp/lora.safetensors");
  EXPECT_FLOAT_EQ(*req.lora_scale, 0.75F);
  EXPECT_EQ(*req.image_return_format, "PNG");
  EXPECT_EQ(*req.image_quality, 95);
}

TEST(ImageGenerateRequestParseTest, CropCoordsTwoElementArrayParsed) {
  auto json = minimalGenerateJson();
  Json::Value arr(Json::arrayValue);
  arr.append(64);
  arr.append(128);
  json["crop_coords_top_left"] = arr;

  auto req = ImageGenerateRequest::fromJson(json, TEST_TASK_ID);
  ASSERT_TRUE(req.crop_coords_top_left.has_value());
  EXPECT_EQ(req.crop_coords_top_left->first, 64);
  EXPECT_EQ(req.crop_coords_top_left->second, 128);
}

TEST(ImageGenerateRequestParseTest, CropCoordsWrongLengthRejected) {
  auto json = minimalGenerateJson();
  Json::Value arr(Json::arrayValue);
  arr.append(1);
  arr.append(2);
  arr.append(3);
  json["crop_coords_top_left"] = arr;

  EXPECT_THROW(ImageGenerateRequest::fromJson(json, TEST_TASK_ID),
               std::invalid_argument);
}

TEST(ImageGenerateRequestParseTest, CropCoordsNonArrayRejected) {
  auto json = minimalGenerateJson();
  json["crop_coords_top_left"] = 42;
  EXPECT_THROW(ImageGenerateRequest::fromJson(json, TEST_TASK_ID),
               std::invalid_argument);
}

TEST(ImageGenerateRequestParseTest, TimestepsAndSigmasArrays) {
  auto json = minimalGenerateJson();
  Json::Value timesteps(Json::arrayValue);
  timesteps.append(1.0);
  timesteps.append(2.5);
  timesteps.append(3.75);
  json["timesteps"] = timesteps;
  Json::Value sigmas(Json::arrayValue);
  sigmas.append(0.1);
  sigmas.append(0.5);
  json["sigmas"] = sigmas;

  auto req = ImageGenerateRequest::fromJson(json, TEST_TASK_ID);
  ASSERT_TRUE(req.timesteps.has_value());
  ASSERT_EQ(req.timesteps->size(), 3U);
  EXPECT_FLOAT_EQ(req.timesteps->at(0), 1.0F);
  EXPECT_FLOAT_EQ(req.timesteps->at(1), 2.5F);
  EXPECT_FLOAT_EQ(req.timesteps->at(2), 3.75F);
  ASSERT_TRUE(req.sigmas.has_value());
  ASSERT_EQ(req.sigmas->size(), 2U);
  EXPECT_FLOAT_EQ(req.sigmas->at(1), 0.5F);
}

TEST(ImageGenerateRequestParseTest, ImageEditFieldsParsed) {
  auto json = minimalGenerateJson("repaint");
  json["image"] = "BASE64_IMAGE_DATA";
  json["mask"] = "BASE64_MASK_DATA";
  json["strength"] = 0.85;

  auto req = ImageGenerateRequest::fromJson(json, TEST_TASK_ID);
  ASSERT_TRUE(req.image.has_value());
  EXPECT_EQ(*req.image, "BASE64_IMAGE_DATA");
  ASSERT_TRUE(req.mask.has_value());
  EXPECT_EQ(*req.mask, "BASE64_MASK_DATA");
  ASSERT_TRUE(req.strength.has_value());
  EXPECT_FLOAT_EQ(*req.strength, 0.85F);
}

TEST(ImageGenerateRequestParseTest, NonStringPromptRejected) {
  Json::Value json;
  json["prompt"] = 123;
  EXPECT_THROW(ImageGenerateRequest::fromJson(json, TEST_TASK_ID),
               std::invalid_argument);
}

TEST(ImageGenerateRequestParseTest, NonIntegerStepsRejected) {
  auto json = minimalGenerateJson();
  json["num_inference_steps"] = "twenty";
  EXPECT_THROW(ImageGenerateRequest::fromJson(json, TEST_TASK_ID),
               std::invalid_argument);
}
