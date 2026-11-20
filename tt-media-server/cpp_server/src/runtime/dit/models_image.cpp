// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <filesystem>
#include <fstream>

#include "runtime/dit/dit_env.hpp"
#include "runtime/dit/dit_models.hpp"

namespace tt::dit {

namespace {

constexpr long kDefaultTraceRegionBytes = 34'541'598;  // Python Settings
constexpr double kKontextGuidanceScale = 3.5;          // BFL default

}  // namespace

long settingsTraceRegion(const std::string& device, const std::string& runner) {
  if (const auto v = envString("TRACE_REGION_SIZE"); v && !v->empty()) {
    return std::stol(*v);
  }
  // FLUX.1 on P300 cards.
  const std::string d = lowered(device);
  if ((runner == "tt-flux.1-dev" || runner == "tt-flux.1-schnell" ||
       runner == "tt-flux.1-kontext-dev") &&
      (d == "p300" || d == "p300x2")) {
    return 51'000'000;
  }
  return kDefaultTraceRegionBytes;
}

ImageModel::ImageModel(RunnerConfig config, ModelSpec spec, DeviceParams device,
                       std::string adapter)
    : DiTModel(std::move(config), std::move(spec)),
      device_(device),
      adapter_(std::move(adapter)) {}

Json::Value ImageModel::adapterOptions(const HostEnv&) const {
  Json::Value o;
  o["checkpoint"] = checkpoint();
  return o;
}

Json::Value ImageModel::buildCall(const DiTRequest& r) const {
  Json::Value call;
  call["prompt"] = r.prompt;
  call["negative_prompt"] =
      r.negativePrompt ? Json::Value(*r.negativePrompt) : Json::Value();
  call["num_inference_steps"] = steps(r);
  call["seed"] = static_cast<Json::Int64>(seed(r));
  return call;
}

FluxKontextModel::FluxKontextModel(RunnerConfig config, ModelSpec spec,
                                   DeviceParams device)
    : ImageModel(std::move(config), std::move(spec), device, "flux_kontext") {}

Json::Value FluxKontextModel::adapterOptions(const HostEnv& host) const {
  Json::Value o = ImageModel::adapterOptions(host);
  // A LoRA applied through LORA_DIR/active_lora.json is fused in when the
  // pipeline is built.
  const std::filesystem::path state =
      std::filesystem::path(envString("LORA_DIR").value_or("/loras")) /
      "active_lora.json";
  std::error_code ec;
  if (std::filesystem::is_regular_file(state, ec)) {
    std::ifstream in(state);
    Json::Value active;
    Json::CharReaderBuilder builder;
    std::string errors;
    if (Json::parseFromStream(builder, in, &active, &errors) &&
        active.isObject()) {
      const std::string path = active.get("path", "").asString();
      if (!path.empty() && std::filesystem::is_regular_file(path, ec)) {
        o["lora_path"] = path;
        o["lora_scale"] = active.get("scale", 1.0).asDouble();
      }
    }
  }
  return o;
}

Json::Value FluxKontextModel::buildCall(const DiTRequest& r) const {
  // Both or neither: mixing one requested axis with a default would skew the
  // aspect ratio. The pipeline snaps to the nearest Kontext bucket.
  const bool sized = r.width && r.height;
  Json::Value call;
  call["prompts"].append(r.prompt);
  // null: the adapter uses the pipeline's own size (1024).
  call["width"] = sized ? Json::Value(*r.width) : Json::Value();
  call["height"] = sized ? Json::Value(*r.height) : Json::Value();
  call["image"] = r.image ? Json::Value(*r.image) : Json::Value();
  call["num_inference_steps"] = steps(r);
  call["guidance_scale"] = kKontextGuidanceScale;
  call["seed"] = static_cast<Json::Int64>(seed(r));
  call["traced"] = traced();
  return call;
}

}  // namespace tt::dit
