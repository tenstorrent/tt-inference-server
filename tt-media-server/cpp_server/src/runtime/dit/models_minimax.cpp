// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <cctype>
#include <cmath>
#include <filesystem>
#include <set>

#include "runtime/dit/dit_env.hpp"
#include "runtime/dit/dit_models.hpp"

namespace tt::dit {

namespace {

// The served envelope; the model's own policy module in tt-metal owns it
// (models/tt_dit/pipelines/minimax_h3/policy.py).
constexpr int kAspectRatios[][2] = {{21, 9}, {16, 9}, {4, 3},
                                    {1, 1},  {3, 4},  {9, 16}};
constexpr int kDefaultDurationS = 5;
constexpr int kMinDurationS = 4;
constexpr int kMaxDurationS = 15;
constexpr int kFps = 24;
constexpr int kFramesPerChunk = 17;
constexpr int kLatentsPerChunk = 5;

constexpr long kTraceRegionBytes = 1'175'000'000;
constexpr int kL1SmallBytesBh = 65536;
constexpr int kL1SmallBytesWh = 32768;
constexpr int kMaxPayloadBytesBh = 8192;
constexpr int kMaxPayloadBytesWh = 4096;

std::string supportedRatios() {
  std::string s;
  for (const auto& r : kAspectRatios) {
    s += (s.empty() ? "" : ", ") + std::to_string(r[0]) + ":" +
         std::to_string(r[1]);
  }
  return s;
}

/** "16:9" (or "16x9", "16/9") -> {16, 9}, restricted to the served set. */
std::pair<int, int> parseAspectRatio(const std::string& value) {
  std::string text = trimmed(value);
  for (char& c : text) {
    if (c == 'x' || c == '/') c = ':';
  }
  const auto colon = text.find(':');
  auto digits = [](const std::string& s) {
    const std::string t = trimmed(s);
    // At most 6 digits: a longer number cannot be a served ratio and would
    // make std::stoi throw something other than RequestError.
    return !t.empty() && t.size() <= 6 &&
           t.find_first_not_of("0123456789") == std::string::npos;
  };
  if (colon == std::string::npos ||
      text.find(':', colon + 1) != std::string::npos ||
      !digits(text.substr(0, colon)) || !digits(text.substr(colon + 1))) {
    throw RequestError("aspect_ratio must look like 'W:H' (got '" + value +
                       "'); supported: " + supportedRatios());
  }
  const int w = std::stoi(text.substr(0, colon));
  const int h = std::stoi(text.substr(colon + 1));
  for (const auto& r : kAspectRatios) {
    if (r[0] == w && r[1] == h) return {w, h};
  }
  throw RequestError("aspect_ratio " + std::to_string(w) + ":" +
                     std::to_string(h) +
                     " is not served; supported: " + supportedRatios());
}

/** Frames for `seconds` at 24 fps, snapped up to the next 17n + 5 the video
 *  VAE can encode. */
int numFramesFor(int seconds) {
  int frames = seconds * kFps;
  while (frames % kFramesPerChunk != kLatentsPerChunk) ++frames;
  return frames;
}

/** A base64 string as the adapter will decode it: the standard alphabet,
 *  padding only at the end, line breaks ignored. */
bool looksLikeBase64(const std::string& text) {
  size_t size = 0, padding = 0;
  for (char c : text) {
    if (c == '\n' || c == '\r') continue;
    if (c == '=') {
      ++padding;
    } else if (padding > 0 || !(std::isalnum(static_cast<unsigned char>(c)) ||
                                c == '+' || c == '/')) {
      return false;
    }
    ++size;
  }
  return size > 0 && size % 4 == 0 && padding <= 2;
}

/** One references.<kind>[index] entry: an object with a base64 string `b64`.
 *  Everything else is the client's mistake and reaches it as a RequestError,
 *  not as a JSON exception from deep inside the call. */
void requireBase64Source(const Json::Value& source, int index,
                         const char* kind) {
  const std::string where =
      std::string("references.") + kind + "[" + std::to_string(index) + "]";
  if (!source.isObject()) throw RequestError(where + " must be an object");
  const Json::Value& b64 = source["b64"];
  if (b64.isNull() || (b64.isString() && b64.asString().empty())) {
    throw RequestError(where +
                       " needs b64 (URLs are not supported by this server)");
  }
  if (!b64.isString() || !looksLikeBase64(b64.asString())) {
    throw RequestError(where + ".b64 must be a base64 string");
  }
}

}  // namespace

MiniMaxModel::MiniMaxModel(RunnerConfig config, ModelSpec spec)
    : DiTModel(std::move(config), std::move(spec)) {}

DeviceParams MiniMaxModel::deviceParams(const HostEnv& host) const {
  // Derived from the pipeline's per-mesh preset so the two cannot disagree: a
  // Ring topology gets the ring fabric, and trace_denoise (preset, or
  // MINIMAX_H3_TRACE_DENOISE) reserves the trace region.
  const MiniMaxPreset preset =
      host.minimaxPreset ? host.minimaxPreset(mesh()) : MiniMaxPreset{};
  DeviceParams p;
  p.ring = preset.ring;
  p.routerPayloadBytes =
      host.blackhole ? kMaxPayloadBytesBh : kMaxPayloadBytesWh;
  p.l1SmallBytes = l1SmallBytes().value_or(host.blackhole ? kL1SmallBytesBh
                                                          : kL1SmallBytesWh);
  const auto envTrace = envBoolOrNull("MINIMAX_H3_TRACE_DENOISE");
  if (envTrace.value_or(preset.traceDenoise)) {
    p.traceRegionBytes = kTraceRegionBytes;
  }
  return p;
}

Json::Value MiniMaxModel::adapterOptions(const HostEnv& host) const {
  Json::Value o;
  o["pipeline"] = "h3";
  o["task"] = task();
  // A local snapshot directory, or null to let the pipeline read
  // MINIMAX_H3_DIFFUSERS_DIR (the default checkpoint is a repo id).
  std::error_code ec;
  o["weights_dir"] = std::filesystem::is_directory(checkpoint(), ec)
                         ? Json::Value(checkpoint())
                         : Json::Value();
  o["dit_fsdp"] = ditFsdp().value_or(!host.blackhole);
  const auto trace = envBoolOrNull("MINIMAX_H3_TRACE_DENOISE");
  const auto bucket = envBoolOrNull("MINIMAX_H3_BUCKET_DENOISE");
  o["trace_denoise"] = trace ? Json::Value(*trace) : Json::Value();
  o["bucket_denoise"] = bucket ? Json::Value(*bucket) : Json::Value();
  addPipelineOptions(o);
  return o;
}

Json::Value MiniMaxModel::buildCall(const DiTRequest& r) const {
  const auto [ratioW, ratioH] = r.aspectRatio ? parseAspectRatio(*r.aspectRatio)
                                              : std::pair<int, int>{16, 9};
  int seconds = kDefaultDurationS;
  if (r.duration) {
    if (*r.duration != std::floor(*r.duration) || *r.duration < kMinDurationS ||
        *r.duration > kMaxDurationS) {
      throw RequestError("duration must be a whole number of seconds from " +
                         std::to_string(kMinDurationS) + " to " +
                         std::to_string(kMaxDurationS));
    }
    seconds = static_cast<int>(*r.duration);
  }
  Json::Value call;
  call["prompt"] = r.prompt;
  call["num_frames"] = numFramesFor(seconds);
  call["aspect_ratio"].append(ratioW);
  call["aspect_ratio"].append(ratioH);
  call["height"] = r.height ? Json::Value(*r.height) : Json::Value();
  call["width"] = r.width ? Json::Value(*r.width) : Json::Value();
  call["num_inference_steps"] = numInferenceSteps();
  call["seed"] = static_cast<Json::Int64>(seed(r));
  addCallInputs(call, r);
  return call;
}

void MiniMaxFastH3Model::addPipelineOptions(Json::Value& o) const {
  if (const auto v = envString("MINIMAX_H3_LORA_PATH")) o["lora_path"] = *v;
  o["vsa_sparsity"] = 0.9;
}

void MiniMaxFL2VAModel::validate(const DiTRequest& r) const {
  if (r.imagePrompts.size() > 2) {
    throw RequestError(
        "MiniMax-H3 FL2VA accepts at most two image_prompts (frame_pos 0 = "
        "first keyframe, -1 = last keyframe)");
  }
  std::set<int> seen;
  for (const auto& p : r.imagePrompts) {
    if (p.framePos != 0 && p.framePos != -1) {
      throw RequestError(
          "MiniMax-H3 FL2VA image_prompts[].frame_pos must be 0 or -1; got " +
          std::to_string(p.framePos));
    }
    if (!seen.insert(p.framePos).second) {
      throw RequestError("duplicate frame_pos in image_prompts");
    }
  }
}

void MiniMaxFL2VAModel::addCallInputs(Json::Value& call,
                                      const DiTRequest& r) const {
  for (const auto& p : r.imagePrompts) {
    if (p.framePos == 0) call["first_image"] = p.image;
    if (p.framePos == -1) call["last_image"] = p.image;
  }
}

void MiniMaxRef2VAModel::validate(const DiTRequest& r) const {
  const Json::Value& refs = r.references;
  bool any = false;
  for (const char* kind : {"images", "videos", "audios"}) {
    if (!refs.isObject() || !refs[kind].isArray()) continue;
    int index = 0;
    for (const auto& source : refs[kind]) {
      requireBase64Source(source, index++, kind);
      any = true;
    }
  }
  if (!any) {
    throw RequestError(
        "MiniMax-H3 Ref2VA needs references (images, videos and/or audios)");
  }
}

void MiniMaxRef2VAModel::addCallInputs(Json::Value& call,
                                       const DiTRequest& r) const {
  // Packed in this order: images, then videos, then audios.
  for (const char* kind : {"images", "videos", "audios"}) {
    Json::Value list(Json::arrayValue);
    if (r.references.isObject() && r.references[kind].isArray()) {
      for (const auto& source : r.references[kind]) {
        list.append(source["b64"]);
      }
    }
    call["references"][kind] = list;
  }
}

void addLightX2VOptions(Json::Value& options, const std::string& modelName,
                        double videoShift, double audioShift) {
  const auto lora = envString("MINIMAX_H3_LORA_PATH");
  std::error_code ec;
  if (!lora || !std::filesystem::is_regular_file(*lora, ec)) {
    throw std::invalid_argument(
        modelName +
        " needs MINIMAX_H3_LORA_PATH set to the lightx2v Turbo adapter's "
        ".safetensors file; got '" +
        lora.value_or("") + "'");
  }
  options["lora_path"] = *lora;
  options["video_shift"] = envString("MINIMAX_H3_VIDEO_SHIFT")
                               ? std::stod(*envString("MINIMAX_H3_VIDEO_SHIFT"))
                               : videoShift;
  options["audio_shift"] = envString("MINIMAX_H3_AUDIO_SHIFT")
                               ? std::stod(*envString("MINIMAX_H3_AUDIO_SHIFT"))
                               : audioShift;
}

}  // namespace tt::dit
