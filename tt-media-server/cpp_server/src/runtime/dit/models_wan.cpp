// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/device_params.hpp"
#include "runtime/dit/dit_models.hpp"

namespace tt::dit {

namespace {

Json::Value imagePromptsJson(const std::vector<ImagePrompt>& prompts) {
  Json::Value list(Json::arrayValue);
  for (const auto& p : prompts) {
    Json::Value item;
    item["image"] = p.image;
    item["frame_pos"] = p.framePos;
    list.append(item);
  }
  return list;
}

}  // namespace

std::pair<int, int> wan22Resolution(MeshShape mesh) {
  return isLargeMesh(mesh) ? std::make_pair(720, 1280)
                           : std::make_pair(480, 832);
}

WanModel::WanModel(RunnerConfig config, ModelSpec spec)
    : DiTModel(std::move(config), std::move(spec)) {
  std::tie(height_, width_) = wan22Resolution(mesh());
}

Json::Value WanModel::adapterOptions(const HostEnv&) const {
  Json::Value o;
  o["checkpoint"] = checkpoint();
  o["height"] = height_;
  o["width"] = width_;
  o["num_frames"] = kWan22NumFrames;
  return o;
}

// Fields the Wan2.2 pipelines do not read (`references`, and `image_prompts` on
// the text-to-video runner) are ignored, like any other extra field of a
// request. The Python server does the same: only its image-to-video route
// takes images.
Json::Value WanModel::buildCall(const DiTRequest& r) const {
  Json::Value call;
  call["prompts"].append(r.prompt);
  call["num_inference_steps"] = steps(r);
  call["guidance_scale"] = 4.0;
  call["guidance_scale_2"] = 3.0;
  call["seed"] = static_cast<Json::Int64>(seed(r));
  call["traced"] = traced();
  if (r.negativePrompt && !r.negativePrompt->empty()) {
    call["negative_prompts"].append(*r.negativePrompt);
  }
  return call;
}

DeviceParams WanModel::deviceParams(const HostEnv& host) const {
  return wan22DeviceParams(mesh(), host.blackhole);
}

Json::Value WanI2VModel::buildCall(const DiTRequest& r) const {
  Json::Value call = WanModel::buildCall(r);
  call["image_prompts"] = imagePromptsJson(r.imagePrompts);
  return call;
}

}  // namespace tt::dit
