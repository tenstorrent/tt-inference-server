// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/video_generate_request.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <stdexcept>
#include <string>

#include "domain/video/video_job.hpp"
#include "domain/video/video_mode.hpp"
#include "domain/video/video_response.hpp"

using tt::domain::VideoGenerateRequest;
using tt::domain::video::VideoResponse;

namespace {

constexpr uint32_t TEST_TASK_ID = 7;

Json::Value promptJson(const std::string& prompt = "a mountain stream") {
  Json::Value json;
  json["prompt"] = prompt;
  return json;
}

void expectReject(const Json::Value& json, const char* message) {
  try {
    VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
    FAIL() << "expected reject: " << message;
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(std::string(e.what()), message);
  }
}

}  // namespace

TEST(VideoGenerateRequestParseTest, MinimalJsonLeavesShapeUnset) {
  auto req = VideoGenerateRequest::fromJson(promptJson(), TEST_TASK_ID);

  EXPECT_EQ(req.task_id, TEST_TASK_ID);
  EXPECT_EQ(req.prompt, "a mountain stream");
  EXPECT_FALSE(req.negativePrompt.has_value());
  // Python resolves missing steps to the default (20).
  ASSERT_TRUE(req.numInferenceSteps.has_value());
  EXPECT_EQ(*req.numInferenceSteps, 20);
  EXPECT_FALSE(req.seed.has_value());
  EXPECT_FALSE(req.aspectRatio.has_value());
  EXPECT_FALSE(req.height.has_value());
  EXPECT_FALSE(req.width.has_value());
  EXPECT_FALSE(req.fps.has_value());
  EXPECT_FALSE(req.duration.has_value());
  EXPECT_FALSE(req.numFrames.has_value());
}

TEST(VideoGenerateRequestParseTest, ExplicitNullStaysUnset) {
  auto json = promptJson();
  json["seed"] = Json::Value::null;
  json["num_inference_steps"] = Json::Value::null;
  json["height"] = Json::Value::null;

  auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);

  EXPECT_FALSE(req.seed.has_value());
  ASSERT_TRUE(req.numInferenceSteps.has_value());
  EXPECT_EQ(*req.numInferenceSteps, 20);
  EXPECT_FALSE(req.height.has_value());
}

TEST(VideoGenerateRequestParseTest, ExplicitFieldsRoundTrip) {
  auto json = promptJson("flowing water");
  json["negative_prompt"] = "blurry";
  json["num_inference_steps"] = 20;
  json["seed"] = 42;
  json["aspect_ratio"] = "16:9";
  json["height"] = 480;
  json["width"] = 832;
  json["fps"] = 16.0;
  json["duration"] = 5.0;
  json["num_frames"] = 81;

  auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
  auto again = VideoGenerateRequest::fromJson(req.toJson(), TEST_TASK_ID);

  EXPECT_EQ(again.prompt, "flowing water");
  ASSERT_TRUE(again.negativePrompt.has_value());
  EXPECT_EQ(*again.negativePrompt, "blurry");
  ASSERT_TRUE(again.numInferenceSteps.has_value());
  EXPECT_EQ(*again.numInferenceSteps, 20);
  ASSERT_TRUE(again.seed.has_value());
  EXPECT_EQ(*again.seed, 42);
  ASSERT_TRUE(again.aspectRatio.has_value());
  EXPECT_EQ(*again.aspectRatio, "16:9");
  ASSERT_TRUE(again.height.has_value());
  EXPECT_EQ(*again.height, 480);
  ASSERT_TRUE(again.width.has_value());
  EXPECT_EQ(*again.width, 832);
  ASSERT_TRUE(again.fps.has_value());
  EXPECT_FLOAT_EQ(*again.fps, 16.0F);
  ASSERT_TRUE(again.duration.has_value());
  EXPECT_FLOAT_EQ(*again.duration, 5.0F);
  ASSERT_TRUE(again.numFrames.has_value());
  EXPECT_EQ(*again.numFrames, 81);
}

TEST(VideoGenerateRequestParseTest, MissingOrBlankPromptIsRejected) {
  expectReject(Json::Value(Json::objectValue),
               "Missing required field: prompt");
  expectReject(promptJson("   "), "prompt must not be empty");
  expectReject(promptJson("\t\n\r"), "prompt must not be empty");
}

TEST(VideoGenerateRequestParseTest, StepBoundsAreAccepted) {
  for (int steps : {4, 50}) {
    auto json = promptJson();
    json["num_inference_steps"] = steps;
    auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
    EXPECT_EQ(*req.numInferenceSteps, steps);
  }
}

TEST(VideoGenerateRequestParseTest, NonPositiveShapeIsRejected) {
  auto rejectField = [](const char* key, const Json::Value& value,
                        const char* message) {
    auto json = promptJson();
    json[key] = value;
    expectReject(json, message);
  };

  const char* stepsMsg = "num_inference_steps must be between 4 and 50";
  rejectField("num_inference_steps", 0, stepsMsg);
  rejectField("num_inference_steps", -3, stepsMsg);
  rejectField("num_inference_steps", 3, stepsMsg);
  rejectField("num_inference_steps", 51, stepsMsg);
  rejectField("height", 0, "height must be positive");
  rejectField("width", -1, "width must be positive");
  rejectField("fps", 0.0, "fps must be positive");
  rejectField("duration", -2.0, "duration must be positive");
  rejectField("num_frames", 0, "num_frames must be positive");
  rejectField("num_frames", -1, "num_frames must be positive");
  rejectField("aspect_ratio", "", "aspect_ratio must not be empty");
}

TEST(VideoGenerateRequestParseTest, WrongTypesAreRejected) {
  auto json = promptJson();
  json["prompt"] = 1;
  expectReject(json, "prompt must be a string");

  json = promptJson();
  json["height"] = "480";
  expectReject(json, "height must be an integer");

  json = promptJson();
  json["fps"] = "fast";
  expectReject(json, "fps must be a number");

  json = promptJson();
  json["seed"] = "one";
  expectReject(json, "seed must be an integer");
}

TEST(VideoGenerateRequestParseTest, SeedZeroRoundTrips) {
  auto json = promptJson();
  json["seed"] = 0;

  auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
  ASSERT_TRUE(req.seed.has_value());
  EXPECT_EQ(*req.seed, 0);

  auto again = VideoGenerateRequest::fromJson(req.toJson(), TEST_TASK_ID);
  ASSERT_TRUE(again.seed.has_value());
  EXPECT_EQ(*again.seed, 0);
}

TEST(VideoGenerateRequestParseTest,
     ExplicitNullOnEveryOptionalFieldStaysUnset) {
  auto json = promptJson();
  for (const char* key :
       {"negative_prompt", "num_inference_steps", "seed", "aspect_ratio",
        "height", "width", "fps", "duration", "num_frames"}) {
    json[key] = Json::Value::null;
  }

  auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
  EXPECT_FALSE(req.negativePrompt.has_value());
  EXPECT_EQ(req.numInferenceSteps.value_or(0), 20);
  EXPECT_FALSE(req.seed.has_value());
  EXPECT_FALSE(req.aspectRatio.has_value());
  EXPECT_FALSE(req.height.has_value());
  EXPECT_FALSE(req.width.has_value());
  EXPECT_FALSE(req.fps.has_value());
  EXPECT_FALSE(req.duration.has_value());
  EXPECT_FALSE(req.numFrames.has_value());
}

TEST(VideoResponseJsonTest, OmitsGenerationTimeUnlessPositive) {
  VideoResponse response(TEST_TASK_ID);
  response.videoPath = "mock://video/7";

  auto json = response.toOpenaiJson();
  EXPECT_EQ(json["video_path"].asString(), "mock://video/7");
  EXPECT_FALSE(json.isMember("generation_time"));

  response.generationTimeSeconds = 0.016;
  json = response.toOpenaiJson();
  EXPECT_DOUBLE_EQ(json["generation_time"].asDouble(), 0.02);
}

TEST(VideoJobJson,
     RequestParametersHaveEveryPythonFieldWithNullsAndDefaultSteps) {
  using tt::domain::video::JobStatus;
  using tt::domain::video::VideoJob;

  VideoJob job;
  job.taskId = 1;
  job.model = "tt-wan2.2";
  job.status = JobStatus::COMPLETED;
  job.createdAt = 10;
  job.completedAt = 20;
  job.requestParameters["prompt"] = "a mountain stream";
  job.requestParameters["seed"] = 1;
  job.requestParameters["num_frames"] = 81;

  const Json::Value json = job.toPublicJson();
  EXPECT_EQ(json["id"].asString(), "1");
  EXPECT_EQ(json["job_type"].asString(), "video");
  EXPECT_EQ(json["status"].asString(), "completed");
  EXPECT_EQ(json["created_at"].asInt64(), 10);
  EXPECT_EQ(json["completed_at"].asInt64(), 20);
  EXPECT_FALSE(json.isMember("error"));

  const Json::Value& params = json["request_parameters"];
  EXPECT_EQ(params["prompt"].asString(), "a mountain stream");
  EXPECT_EQ(params["seed"].asInt(), 1);
  EXPECT_EQ(params["num_frames"].asInt(), 81);
  EXPECT_EQ(params["num_inference_steps"].asInt(),
            tt::domain::kDefaultVideoInferenceSteps);
  for (const char* unset : {"negative_prompt", "aspect_ratio", "height",
                            "width", "fps", "duration"}) {
    ASSERT_TRUE(params.isMember(unset)) << unset;
    EXPECT_TRUE(params[unset].isNull()) << unset;
  }
}

TEST(VideoModeTest, EnvDefaultsToAsyncAndRequestOverrides) {
  ::unsetenv("USE_ASYNC_VIDEO");
  Json::Value body(Json::objectValue);
  EXPECT_TRUE(tt::domain::video::videoRequestIsAsync("", body));

  body["async"] = false;
  EXPECT_FALSE(tt::domain::video::videoRequestIsAsync("", body));

  body["async"] = true;
  EXPECT_FALSE(tt::domain::video::videoRequestIsAsync("false", body));

  ::setenv("USE_ASYNC_VIDEO", "false", 1);
  body.removeMember("async");
  EXPECT_FALSE(tt::domain::video::videoRequestIsAsync("", body));
  EXPECT_TRUE(tt::domain::video::videoRequestIsAsync("true", body));
  ::unsetenv("USE_ASYNC_VIDEO");
}

TEST(VideoGenerateRequestParseTest, ImagePromptsPassThroughAndAreChecked) {
  Json::Value json = promptJson();
  Json::Value first;
  first["image"] = "aGVsbG8=";
  json["image_prompts"].append(first);
  Json::Value last;
  last["image"] = "d29ybGQ=";
  last["frame_pos"] = -1;
  json["image_prompts"].append(last);

  const auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
  ASSERT_TRUE(req.imagePrompts.has_value());
  ASSERT_EQ(req.imagePrompts->size(), 2u);
  EXPECT_EQ((*req.imagePrompts)[0]["frame_pos"].asInt(), 0);
  EXPECT_EQ((*req.imagePrompts)[1]["image"].asString(), "d29ybGQ=");
  // The worker gets the images; the job record only what they pin.
  EXPECT_EQ(req.toJson()["image_prompts"][1]["image"].asString(), "d29ybGQ=");
  EXPECT_FALSE(
      req.toJson(/*withMedia=*/false)["image_prompts"][1].isMember("image"));
  EXPECT_EQ(req.toJson(false)["image_prompts"][1]["frame_pos"].asInt(), -1);

  Json::Value duplicate = json;
  duplicate["image_prompts"][1]["frame_pos"] = 0;
  expectReject(duplicate, "duplicate frame_pos=0 in image_prompts");
  Json::Value outOfRange = json;
  outOfRange["image_prompts"][1]["frame_pos"] = 81;
  expectReject(outOfRange, "image_prompts[].frame_pos must be in [-1, 81)");
  Json::Value empty = promptJson();
  empty["image_prompts"] = Json::arrayValue;
  expectReject(empty, "image_prompts must have 1 to 81 entries");
  Json::Value noImage = promptJson();
  noImage["image_prompts"].append(Json::objectValue);
  expectReject(noImage, "image_prompts[].image must not be empty");
}

TEST(VideoGenerateRequestParseTest, ReferencesTakeBase64Only) {
  Json::Value json = promptJson();
  Json::Value image;
  image["b64"] = "aGVsbG8=";
  json["references"]["images"].append(image);
  const auto req = VideoGenerateRequest::fromJson(json, TEST_TASK_ID);
  ASSERT_TRUE(req.references.has_value());
  EXPECT_EQ(req.toJson()["references"]["images"][0]["b64"].asString(),
            "aGVsbG8=");
  EXPECT_EQ(req.toJson(false)["references"]["images"].asUInt(), 1u);
  EXPECT_EQ(req.toJson(false)["references"]["audios"].asUInt(), 0u);

  Json::Value url = promptJson();
  url["references"]["videos"][0]["url"] = "https://example.com/a.mp4";
  expectReject(url,
               "references.videos[].url is not supported by this server; send "
               "b64");
  Json::Value unknown = promptJson();
  unknown["references"]["masks"] = Json::arrayValue;
  expectReject(unknown,
               "references.masks is not supported; use images, videos or "
               "audios");
  Json::Value none = promptJson();
  none["references"] = Json::objectValue;
  expectReject(none, "references must contain at least one item");
}
