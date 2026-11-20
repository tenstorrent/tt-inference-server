// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/device_params.hpp"
#include "runtime/dit/dit_env.hpp"
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

int WanModel::stepsFor(const DiTRequest& request) const {
  return fixedSteps() > 0 ? fixedSteps() : DiTModel::stepsFor(request);
}

Json::Value WanModel::adapterOptions(const HostEnv&) const {
  Json::Value o;
  o["checkpoint"] = checkpoint();
  o["height"] = height_;
  o["width"] = width_;
  o["num_frames"] = kWan22NumFrames;
  return o;
}

Json::Value WanModel::buildCall(const DiTRequest& r) const {
  Json::Value call;
  call["prompts"].append(r.prompt);
  call["num_inference_steps"] = steps(r);
  call["guidance_scale"] = guidanceScale();
  call["guidance_scale_2"] = guidanceScale2();
  call["seed"] = static_cast<Json::Int64>(seed(r));
  call["traced"] = traced();
  if (usesNegativePrompt() && r.negativePrompt && !r.negativePrompt->empty()) {
    call["negative_prompts"].append(*r.negativePrompt);
  }
  return call;
}

DeviceParams WanModel::deviceParams(const HostEnv& host) const {
  DeviceParams p = wan22DeviceParams(mesh(), host.blackhole);
  if (largeBhTraceRegion() > 0 && isLargeMesh(mesh()) && host.blackhole) {
    p.traceRegionBytes = largeBhTraceRegion();
  }
  return p;
}

Json::Value WanI2VModel::buildCall(const DiTRequest& r) const {
  Json::Value call = WanModel::buildCall(r);
  call["image_prompts"] = imagePromptsJson(r.imagePrompts);
  return call;
}

std::vector<std::pair<std::string, std::string>> WanAniSoraModel::envDefaults()
    const {
  // Build-time switches of the fast image-encode path (about 7.5 s -> 0.35 s of
  // image encode); a deployment can still pin any of them to "0".
  return {{"WAN_ANISORA_FAST_VAE_ENCODER", "1"},
          {"WAN_ANISORA_ENCODER_T_OUT_1", "1"},
          {"WAN_ANISORA_ONDEVICE_COND", "1"}};
}

std::vector<std::pair<std::string, std::string>> WanDistillModel::envDefaults()
    const {
  return {{"WAN_DISTILL_FAST_VAE_ENCODER", "1"},
          {"WAN_DISTILL_ENCODER_T_OUT_1", "1"},
          {"WAN_DISTILL_ONDEVICE_COND", "1"}};
}

Json::Value WanLoraModel::adapterOptions(const HostEnv& host) const {
  Json::Value o = WanModel::adapterOptions(host);
  if (const auto v = envString("LORA_HIGH_PATH")) o["lora_high"] = *v;
  if (const auto v = envString("LORA_LOW_PATH")) o["lora_low"] = *v;
  return o;
}

Json::Value WanLightningModel::adapterOptions(const HostEnv& host) const {
  Json::Value o = WanLoraModel::adapterOptions(host);
  o["cfg_enabled"] = false;
  o["boundary_ratio"] = 0.875;
  // The adapter downloads the adapters from this repo for the paths not set.
  o["download_repo"] = "lightx2v/Wan2.2-Lightning";
  o["download_high"] =
      "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors";
  o["download_low"] =
      "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors";
  return o;
}

Json::Value WanProdiaModel::buildCall(const DiTRequest& r) const {
  Json::Value call;
  call["prompt"] = r.prompt;
  call["height"] = height_;
  call["width"] = width_;
  call["num_frames"] = kWan22NumFrames;
  call["seed"] = static_cast<Json::Int64>(seed(r));
  call["traced"] = traced();
  return call;
}

DeviceParams WanProdiaModel::deviceParams(const HostEnv& host) const {
  return prodiaDeviceParams(mesh(), host.blackhole);
}

Json::Value WanProdiaI2VModel::buildCall(const DiTRequest& r) const {
  Json::Value call = WanProdiaModel::buildCall(r);
  // The adapter resizes each image to the output size.
  call["image_prompts"] = imagePromptsJson(r.imagePrompts);
  return call;
}

}  // namespace tt::dit
