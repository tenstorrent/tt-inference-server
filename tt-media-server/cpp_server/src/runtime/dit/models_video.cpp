// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <cmath>
#include <filesystem>

#include "runtime/dit/device_params.hpp"
#include "runtime/dit/dit_models.hpp"
#include "runtime/dit/media_export.hpp"

namespace tt::dit {

namespace {

constexpr int kLtxTemporalCompression = 8;
constexpr int kLtxL1SmallBytes = 32768;
constexpr long kLtxTraceRegionBytes = 500'000'000;
constexpr double kMochiGuidanceScale = 3.5;

}  // namespace

// -- Mochi ------------------------------------------------------------------

MochiModel::MochiModel(RunnerConfig config)
    : DiTModel(std::move(config),
               ModelSpec{.logName = "Mochi1",
                         .meshes = {{"t3k", {2, 4}},
                                    {"galaxy", {4, 8}},
                                    {"p150x4", {1, 4}},
                                    {"p150x8", {2, 4}},
                                    {"p300x2", {2, 2}},
                                    {"galaxy_quad", {4, 32}}},
                         .checkpoint = "genmo/mochi-1-preview",
                         .checkpointFromConfig = false}) {}

Json::Value MochiModel::adapterOptions(const HostEnv&) const {
  Json::Value o;
  o["checkpoint"] = checkpoint();
  return o;
}

Json::Value MochiModel::buildCall(const DiTRequest& r) const {
  Json::Value call;
  call["prompts"].append(r.prompt);
  call["negative_prompts"].append(r.negativePrompt.value_or(""));
  call["num_inference_steps"] = steps(r);
  call["guidance_scale"] = kMochiGuidanceScale;
  call["seed"] = seed(r);
  return call;
}

// -- LTX --------------------------------------------------------------------

int ltxSnapNumFrames(long n) {
  if (n < 1) return 1;
  const long k =
      std::lround(static_cast<double>(n - 1) / kLtxTemporalCompression);
  return static_cast<int>(k * kLtxTemporalCompression + 1);
}

LtxModel::LtxModel(RunnerConfig config)
    : DiTModel(std::move(config),
               ModelSpec{.logName = "LTX-2.3-distilled",
                         .meshes = {{"galaxy", {4, 8}}},
                         .checkpoint =
                             "Lightricks/LTX-2.3:ltx-2.3-22b-distilled-1.1."
                             "safetensors",
                         .defaultFps = kLtxFps}) {}

Json::Value LtxModel::adapterOptions(const HostEnv&) const {
  Json::Value o;
  o["checkpoint"] = checkpoint();
  o["traced"] = traced();
  o["num_frames"] = kLtxNumFrames;
  o["height"] = kLtxHeight;
  o["width"] = kLtxWidth;
  o["fps"] = kLtxFps;
  return o;
}

DeviceParams LtxModel::deviceParams(const HostEnv& host) const {
  // Wan2.2's ring fabric, plus L1_SMALL for the traced decode and vocoder and
  // the larger trace region of the two stage traces.
  DeviceParams p = wan22DeviceParams(mesh(), host.blackhole);
  p.l1SmallBytes = kLtxL1SmallBytes;
  if (traced()) p.traceRegionBytes = kLtxTraceRegionBytes;
  return p;
}

void LtxModel::validate(const DiTRequest& r) const {
  // num_inference_steps is not checked: the server fills in its default and
  // this model takes no step count (a fixed 8 + 3 schedule).
  const std::string detail = "this deployment serves " +
                             std::to_string(kLtxNumFrames) + " frames at " +
                             std::to_string(kLtxHeight) + "x" +
                             std::to_string(kLtxWidth) + ", 25 fps";
  if (r.aspectRatio) {
    throw RequestError(
        "aspect_ratio is not supported by LTX-2.3; use height/width");
  }
  if (r.fps && *r.fps != kLtxFps) {
    throw RequestError("fps=" + std::to_string(*r.fps) + " is not served; " +
                       detail);
  }
  if (r.height && *r.height != kLtxHeight) {
    throw RequestError("height=" + std::to_string(*r.height) +
                       " is not served; " + detail);
  }
  if (r.width && *r.width != kLtxWidth) {
    throw RequestError("width=" + std::to_string(*r.width) +
                       " is not served; " + detail);
  }
  std::optional<int> frames = r.numFrames;
  if (r.duration) {
    const int fromDuration =
        ltxSnapNumFrames(std::lround(*r.duration * kLtxFps));
    if (frames && *frames != fromDuration) {
      throw RequestError(
          "duration implies num_frames=" + std::to_string(fromDuration) +
          ", which contradicts num_frames=" + std::to_string(*frames));
    }
    frames = fromDuration;
  }
  if (frames && *frames != kLtxNumFrames) {
    throw RequestError("num_frames=" + std::to_string(*frames) +
                       " is not served; " + detail);
  }
}

Json::Value LtxModel::buildCall(const DiTRequest& r) const {
  Json::Value call;
  call["prompt"] = r.prompt;
  call["output_path"] =
      (std::filesystem::path(config().outputDir) / (uuid4() + ".mp4")).string();
  call["num_frames"] = kLtxNumFrames;
  call["height"] = kLtxHeight;
  call["width"] = kLtxWidth;
  call["seed"] = seed(r);
  call["fps"] = kLtxFps;
  return call;
}

}  // namespace tt::dit
