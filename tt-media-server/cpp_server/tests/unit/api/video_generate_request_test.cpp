// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/video_generate_request.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/video/video_deployment.hpp"
#include "domain/video/video_job.hpp"
#include "domain/video/video_media.hpp"
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
  expectReject(
      duplicate,
      "duplicate frame_pos=0 in image_prompts; each image must target a "
      "distinct frame");
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
  // Python ignores a group it does not know, so a request with only that is
  // empty
  Json::Value unknown = promptJson();
  unknown["references"]["masks"] = Json::arrayValue;
  expectReject(unknown,
               "ref2va needs at least one reference; use a t2va call for "
               "text-only requests");
  Json::Value none = promptJson();
  none["references"] = Json::objectValue;
  expectReject(none,
               "ref2va needs at least one reference; use a t2va call for "
               "text-only requests");
  // exactly one of b64 and url, as in Python's MediaSource
  Json::Value both = promptJson();
  both["references"]["images"][0]["b64"] = "aGVsbG8=";
  both["references"]["images"][0]["url"] = "https://example.com/a.png";
  expectReject(both, "provide exactly one of b64 or url");
  Json::Value neither = promptJson();
  neither["references"]["images"][0] = Json::objectValue;
  expectReject(neither, "provide exactly one of b64 or url");
  // a group Python does not know is ignored next to a real one
  Json::Value extra = promptJson();
  extra["references"]["images"][0]["b64"] = "aGVsbG8=";
  extra["references"]["masks"][0]["b64"] = "aGVsbG8=";
  EXPECT_NO_THROW(VideoGenerateRequest::fromJson(extra, TEST_TASK_ID));
}

// ---- image-to-video: routes, media rules ---------------------------

namespace media_test {

using namespace tt::domain::video;

std::string toBase64(const std::vector<uint8_t>& bytes) {
  static const char* alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t chunk =
        (uint32_t(bytes[i]) << 16) |
        (i + 1 < bytes.size() ? uint32_t(bytes[i + 1]) << 8 : 0) |
        (i + 2 < bytes.size() ? uint32_t(bytes[i + 2]) : 0);
    out += alphabet[(chunk >> 18) & 63];
    out += alphabet[(chunk >> 12) & 63];
    out += i + 1 < bytes.size() ? alphabet[(chunk >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? alphabet[chunk & 63] : '=';
  }
  return out;
}

/** The first bytes of a PNG: signature and IHDR with the given size. */
std::vector<uint8_t> pngHeader(uint32_t width, uint32_t height) {
  std::vector<uint8_t> b = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
                            0,    0,   0,   13,  'I',  'H',  'D',  'R'};
  for (uint32_t v : {width, height}) {
    b.push_back(uint8_t(v >> 24));
    b.push_back(uint8_t(v >> 16));
    b.push_back(uint8_t(v >> 8));
    b.push_back(uint8_t(v));
  }
  b.insert(b.end(), {8, 2, 0, 0, 0});
  return b;
}

Json::Value requestWithPrompts(
    const std::vector<std::pair<std::string, int>>& list) {
  Json::Value json;
  json["prompt"] = "a stream";
  for (const auto& [image, framePos] : list) {
    Json::Value entry;
    entry["image"] = image;
    entry["frame_pos"] = framePos;
    json["image_prompts"].append(entry);
  }
  return json;
}

int statusOf(const VideoGenerateRequest& request, VideoDeployment d,
             std::string* message = nullptr) {
  try {
    checkRequestMedia(request, d);
  } catch (const MediaRequestError& e) {
    if (message != nullptr) *message = e.what();
    return e.status();
  }
  return 0;
}

}  // namespace media_test

TEST(VideoDeploymentTest, RunnerNamesAreClassifiedLikePython) {
  using namespace tt::domain::video;
  EXPECT_EQ(classifyVideoRunner("tt-wan2.2"), VideoDeployment::Text);
  EXPECT_EQ(classifyVideoRunner(""), VideoDeployment::Text);
  EXPECT_EQ(classifyVideoRunner("tt-wan2.2-i2v"), VideoDeployment::ImageOnly);
  EXPECT_EQ(classifyVideoRunner("TT-WAN2.2-I2V"), VideoDeployment::ImageOnly);
}

TEST(VideoDeploymentTest, RoutesAreRefusedWherePythonRefusesThem) {
  using namespace tt::domain::video;
  // text-only route: refused on the image-conditioned deployment
  EXPECT_FALSE(routeRefusal(VideoRoute::Text, VideoDeployment::Text));
  EXPECT_TRUE(routeRefusal(VideoRoute::Text, VideoDeployment::ImageOnly));
  // i2v route: allowed on Wan (also the T2V runner, which ignores the image)
  EXPECT_FALSE(routeRefusal(VideoRoute::I2V, VideoDeployment::Text));
  EXPECT_FALSE(routeRefusal(VideoRoute::I2V, VideoDeployment::ImageOnly));
  EXPECT_NE(routeRefusal(VideoRoute::Text, VideoDeployment::ImageOnly)
                ->find("/generations/i2v"),
            std::string::npos);
}

TEST(VideoMediaTest, Base64DecodeAcceptsLineBreaksAndRejectsGarbage) {
  using namespace tt::domain::video;
  const auto ok = base64Decode("aGVs\nbG8=");
  ASSERT_TRUE(ok.has_value());
  EXPECT_EQ(std::string(ok->begin(), ok->end()), "hello");
  EXPECT_FALSE(base64Decode("").has_value());
  EXPECT_FALSE(base64Decode("aGVs*G8=").has_value());
  EXPECT_FALSE(base64Decode("aGVs=bG8").has_value());   // data after padding
  EXPECT_FALSE(base64Decode("aGVsb").has_value());      // 1 symbol over a group
  EXPECT_FALSE(base64Decode("aGVsbG8==").has_value());  // padding does not fit
}

TEST(VideoMediaTest, Base64DecodeLikePython) {
  using namespace tt::domain::video;
  const auto text = [](const std::optional<std::vector<uint8_t>>& bytes) {
    return bytes ? std::string(bytes->begin(), bytes->end()) : "<none>";
  };
  // Missing padding (JSON transport may strip it).
  EXPECT_EQ(text(base64Decode("aGVsbG8")), "hello");
  EXPECT_EQ(text(base64Decode("aGVsbG8=")), "hello");
  EXPECT_EQ(text(base64Decode("aGk")), "hi");
  // A data URI prefix is skipped.
  EXPECT_EQ(text(base64Decode("data:image/png;base64,aGVsbG8=")), "hello");
  EXPECT_EQ(text(base64Decode("data:image/png;base64,aGVsbG8")), "hello");
  EXPECT_FALSE(base64Decode("data:image/png;base64").has_value());   // no comma
  EXPECT_FALSE(base64Decode("data:image/png;base64,").has_value());  // no data
}

TEST(VideoMediaTest, BmpHeightOfInt32MinIsRefusedNotUndefined) {
  using namespace tt::domain::video;
  std::vector<uint8_t> bmp(26, 0);
  bmp[0] = 'B';
  bmp[1] = 'M';
  bmp[18] = 2;  // width 2
  bmp[22] = 3;  // height 3
  const auto ok = imageSize(bmp);
  ASSERT_TRUE(ok.has_value());
  EXPECT_EQ(*ok, std::make_pair(2, 3));
  bmp[22] = 0xFD;  // height -3: a top-down bitmap
  bmp[23] = bmp[24] = bmp[25] = 0xFF;
  EXPECT_EQ(*imageSize(bmp), std::make_pair(2, 3));
  bmp[22] = bmp[23] = bmp[24] = 0;
  bmp[25] = 0x80;  // height 0x80000000
  EXPECT_FALSE(imageSize(bmp).has_value());
}

TEST(VideoMediaTest, ImageSizeIsReadFromThePngJpegWebpHeaders) {
  using namespace tt::domain::video;
  auto png = imageSize(media_test::pngHeader(832, 480));
  ASSERT_TRUE(png.has_value());
  EXPECT_EQ(*png, std::make_pair(832, 480));

  // SOI, an APP0 segment, then SOF0 with height 6 and width 10
  std::vector<uint8_t> jpeg = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x4A,
                               0x46, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00,
                               0x06, 0x00, 0x0A, 0x03, 0x01, 0x22, 0x00};
  auto j = imageSize(jpeg);
  ASSERT_TRUE(j.has_value());
  EXPECT_EQ(*j, std::make_pair(10, 6));

  // A real lossy WebP of 12x8 (VP8 chunk) written by PIL.
  const auto webp = base64Decode(
      "UklGRjoAAABXRUJQVlA4IC4AAACwAQCdASoMAAgAAUAmJaACdLoABDAAAP7x3I/4DdfFtMv/"
      "vYL/3YL/3YL/WwAA");
  ASSERT_TRUE(webp.has_value());
  auto w = imageSize(*webp);
  ASSERT_TRUE(w.has_value());
  EXPECT_EQ(*w, std::make_pair(12, 8));

  // VP8X (extended) header: canvas size minus one in 24 bits
  std::vector<uint8_t> vp8x = {
      'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E',  'B',  'P',  'V',  'P',  '8',
      'X', 10,  0,   0,   0, 0, 0, 0, 0,   0x1F, 0x03, 0x00, 0xDF, 0x01, 0x00};
  auto x = imageSize(vp8x);
  ASSERT_TRUE(x.has_value());
  EXPECT_EQ(*x, std::make_pair(800, 480));

  EXPECT_FALSE(imageSize({}).has_value());
  EXPECT_FALSE(
      imageSize({'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'a', 'g', 'e'})
          .has_value());
}

TEST(VideoMediaTest, ImagePromptsAreCheckedAtSubmit) {
  using namespace tt::domain::video;
  using media_test::statusOf;
  const std::string good = media_test::toBase64(media_test::pngHeader(64, 64));
  const std::string tiny =
      media_test::toBase64(media_test::pngHeader(100, 100));

  auto parse = [](const Json::Value& json) {
    return VideoGenerateRequest::fromJson(json, 1);
  };
  // Wan: any decodable image is accepted, whatever its size
  EXPECT_EQ(
      statusOf(parse(media_test::requestWithPrompts({{good, 0}, {tiny, 80}})),
               VideoDeployment::ImageOnly),
      0);
  // not an image: 422 and the field named
  std::string message;
  EXPECT_EQ(
      statusOf(parse(media_test::requestWithPrompts({{"bm90IGFuIGltYWdl", 0}})),
               VideoDeployment::ImageOnly, &message),
      422);
  EXPECT_NE(message.find("image_prompts[0] is not a decodable image"),
            std::string::npos);
  EXPECT_EQ(statusOf(parse(media_test::requestWithPrompts({{"!!!", 0}})),
                     VideoDeployment::ImageOnly),
            422);
}

TEST(VideoMediaTest, InlineMediaOverTheCapIs413AndOneImageOverItsCapIs422) {
  using namespace tt::domain::video;
  using media_test::statusOf;
  // Over the per-image cap (40,000,000 base64 chars): 422, named
  Json::Value one = media_test::requestWithPrompts(
      {{std::string(kMaxBase64ImageLen + 4, 'A'), 0}});
  std::string message;
  EXPECT_EQ(statusOf(VideoGenerateRequest::fromJson(one, 1),
                     VideoDeployment::ImageOnly, &message),
            422);
  EXPECT_NE(message.find("exceeds the 40000000-char cap"), std::string::npos);
  // Two images that each fit but together pass 64,000,000 chars: 413
  const std::string big(kMaxBase64ImageLen - 8, 'A');
  Json::Value two = media_test::requestWithPrompts({{big, 0}, {big, 1}});
  EXPECT_EQ(statusOf(VideoGenerateRequest::fromJson(two, 1),
                     VideoDeployment::ImageOnly, &message),
            413);
  EXPECT_NE(message.find("Inline media totals"), std::string::npos);
}

TEST(VideoMediaTest, MediaUrlsAreRecognised) {
  using namespace tt::domain::video;
  EXPECT_TRUE(
      isMediaUrl("https://bucket.s3.amazonaws.com/a.png?X-Amz-Signature=1"));
  EXPECT_TRUE(isMediaUrl("HTTP://example.com/a.png"));
  EXPECT_FALSE(isMediaUrl("iVBORw0KGgo="));
  EXPECT_FALSE(isMediaUrl("ftp://example.com/a.png"));
}
