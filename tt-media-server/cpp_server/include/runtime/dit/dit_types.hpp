// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tt::dit {

/** A request the model cannot serve (the Python server's ValueError). It is
 *  the client's mistake: with several hosts every rank refuses it the same way
 *  before touching the mesh, so the job just fails. */
class RequestError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/** The result could not be written out (ffmpeg missing or failed, disk full)
 *  after the pipeline ran. Not the client's mistake and not a failure of the
 *  runner: the pipeline itself finished. */
class ExportError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct MeshShape {
  int rows = 1;
  int cols = 1;
  int size() const { return rows * cols; }
  bool operator==(const MeshShape& o) const {
    return rows == o.rows && cols == o.cols;
  }
};

/** What the C++ worker knows about the model it serves. */
struct RunnerConfig {
  std::string modelRunner;
  std::string device;
  std::string modelWeightsPath;
  std::string outputDir = "/tmp/videos";
  int workerId = 0;

  /** MODEL_RUNNER, DEVICE, MODEL_WEIGHTS_PATH, TT_VIDEO_OUTPUT_DIR. */
  static RunnerConfig fromEnv();
};

/** Facts about the host that the models need and only tt-metal can answer. */
struct HostEnv {
  bool blackhole = false;
};

/** How the mesh is opened (see openMesh in dit_runner.cpp). Anything unset
 *  keeps tt-metal's default. */
struct DeviceParams {
  bool ring = false;         // FABRIC_1D_RING instead of FABRIC_1D
  bool relaxedInit = false;  // FabricReliabilityMode.RELAXED_INIT
  std::optional<int> routerPayloadBytes;
  std::optional<long> traceRegionBytes;
  std::optional<int> l1SmallBytes;
  bool operator==(const DeviceParams& o) const {
    return ring == o.ring && relaxedInit == o.relaxedInit &&
           routerPayloadBytes == o.routerPayloadBytes &&
           traceRegionBytes == o.traceRegionBytes &&
           l1SmallBytes == o.l1SmallBytes;
  }
};

enum class OutputKind { Video, Image };

struct ImagePrompt {
  std::string image;  // base64
  int framePos = 0;   // -1 is the last frame
};

/** The request a model receives, built from the JSON the server forwards. */
struct DiTRequest {
  std::string prompt;
  std::optional<std::string> negativePrompt;
  std::optional<int> numInferenceSteps;
  std::optional<int> seed;
  std::optional<double> fps;
  std::optional<int> height;
  std::optional<int> width;
  std::optional<int> numFrames;
  std::optional<double> duration;
  std::optional<std::string> aspectRatio;
  std::vector<ImagePrompt> imagePrompts;  // image-to-video, MiniMax keyframes
  Json::Value references;                 // Ref2VA: {images, videos, audios}
  std::optional<std::string> image;       // image editing (base64)
  std::string taskId;

  /** Throws RequestError when `prompt` is missing or the fields have the
   *  wrong type or shape. */
  static DiTRequest fromJson(const Json::Value& json);
  Json::Value toJson() const;
};

/** What generate() returns: the mp4 path (video) or PNG bytes (image). */
struct DiTResult {
  OutputKind kind = OutputKind::Video;
  std::string path;
  std::vector<std::string> images;
};

/** Wan2.2 I2V rules: at least one image, frame_pos in [-1, numFrames), each
 *  frame pinned once. */
void validateImagePrompts(const std::vector<ImagePrompt>& prompts,
                          int numFrames = 81);

}  // namespace tt::dit
