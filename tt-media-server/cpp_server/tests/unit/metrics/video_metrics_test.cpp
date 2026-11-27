// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>

#include "metrics/metrics.hpp"

using tt::metrics::ServerMetrics;

namespace {

// Value of the first sample of `series` in the exposition text; -1 when absent.
double sampleOf(const std::string& text, const std::string& series) {
  const std::string key = "\n" + series + " ";
  const auto at = text.find(key);
  if (at == std::string::npos) return -1;
  return std::stod(text.substr(at + key.size()));
}

}  // namespace

TEST(VideoStageMetrics, EachStageIsRecordedOnItsOwnHistogram) {
  auto& metrics = ServerMetrics::instance();
  const std::string before = "\n" + metrics.renderText();
  const double denoiseBefore =
      std::max(0.0, sampleOf(before, "tt_video_denoise_seconds_count"));
  const double vaeBefore =
      std::max(0.0, sampleOf(before, "tt_video_vae_decode_seconds_count"));

  metrics.onVideoStages(100.0, 10.0);

  const std::string after = "\n" + metrics.renderText();
  EXPECT_EQ(sampleOf(after, "tt_video_denoise_seconds_count"),
            denoiseBefore + 1);
  EXPECT_EQ(sampleOf(after, "tt_video_vae_decode_seconds_count"),
            vaeBefore + 1);
  EXPECT_GE(sampleOf(after, "tt_video_denoise_seconds_sum"), 100.0);
  EXPECT_GE(sampleOf(after, "tt_video_vae_decode_seconds_sum"), 10.0);
}

TEST(VideoStageMetrics, AStageThatWasNotTimedIsSkipped) {
  auto& metrics = ServerMetrics::instance();
  const std::string before = "\n" + metrics.renderText();
  const double denoiseBefore =
      std::max(0.0, sampleOf(before, "tt_video_denoise_seconds_count"));
  const double vaeBefore =
      std::max(0.0, sampleOf(before, "tt_video_vae_decode_seconds_count"));

  metrics.onVideoStages(std::nullopt, 7.0);  // the decode only
  metrics.onVideoStages(std::nullopt, std::nullopt);

  const std::string after = "\n" + metrics.renderText();
  EXPECT_EQ(sampleOf(after, "tt_video_denoise_seconds_count"), denoiseBefore);
  EXPECT_EQ(sampleOf(after, "tt_video_vae_decode_seconds_count"),
            vaeBefore + 1);
}

TEST(VideoOutputMetrics, WhatAVideoAddedUpToIsCounted) {
  auto& metrics = ServerMetrics::instance();
  const auto read = [&](const std::string& series) {
    return std::max(0.0, sampleOf("\n" + metrics.renderText(), series));
  };
  const double frames = read("tt_video_frames_generated_total");
  const double steps = read("tt_video_denoise_steps_total");
  const double bytes = read("tt_video_output_bytes_total");
  const double seconds = read("tt_video_content_seconds_total");
  const double fpsSamples = read("tt_video_frames_per_second_count");

  metrics.onVideoOutput(81, 40, 3000000, 5.0625, 125.0);

  EXPECT_EQ(read("tt_video_frames_generated_total"), frames + 81);
  EXPECT_EQ(read("tt_video_denoise_steps_total"), steps + 40);
  EXPECT_EQ(read("tt_video_output_bytes_total"), bytes + 3000000);
  EXPECT_DOUBLE_EQ(read("tt_video_content_seconds_total"), seconds + 5.0625);
  EXPECT_EQ(read("tt_video_frames_per_second_count"), fpsSamples + 1);
}

TEST(VideoOutputMetrics, WhatIsNotKnownIsSkipped) {
  auto& metrics = ServerMetrics::instance();
  const auto read = [&](const std::string& series) {
    return std::max(0.0, sampleOf("\n" + metrics.renderText(), series));
  };
  const double frames = read("tt_video_frames_generated_total");
  const double bytes = read("tt_video_output_bytes_total");
  const double fpsSamples = read("tt_video_frames_per_second_count");

  // no frame count: no frames, no frames per second; the bytes still count
  metrics.onVideoOutput(std::nullopt, std::nullopt, 1000, std::nullopt, 10.0);
  // no wall time: the frames count, the rate has no sample
  metrics.onVideoOutput(10, std::nullopt, std::nullopt, std::nullopt, 0.0);

  EXPECT_EQ(read("tt_video_frames_generated_total"), frames + 10);
  EXPECT_EQ(read("tt_video_output_bytes_total"), bytes + 1000);
  EXPECT_EQ(read("tt_video_frames_per_second_count"), fpsSamples);
}
