// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// MODEL_RUNNER -> model. One row per runner of the Python server's
// tt_model_runners/dit_runners.py.

#include <algorithm>

#include "runtime/dit/device_params.hpp"
#include "runtime/dit/dit_models.hpp"

namespace tt::dit {

namespace {

using Meshes = std::map<std::string, MeshShape>;

const char* const kWanT2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";
const char* const kWanI2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";
const char* const kH3 = "MiniMaxAI/MiniMax-H3";

const Meshes kWanMeshes = {{"n150", {1, 1}},   {"t3k", {2, 4}},
                           {"galaxy", {4, 8}}, {"bh-galaxy", {4, 8}},
                           {"p150x4", {1, 4}}, {"p150x8", {2, 4}},
                           {"p300x2", {2, 2}}, {"galaxy_quad", {4, 32}}};
Meshes withoutN150(Meshes m) {
  m.erase("n150");
  return m;
}
const Meshes kBhGalaxy = {{"bh-galaxy", {4, 8}}, {"galaxy_quad", {4, 32}}};
const Meshes kProdia = {{"galaxy", {4, 8}}, {"galaxy_quad", {4, 32}}};
const Meshes kH3Meshes = {{"galaxy", {4, 8}}, {"galaxy_quad", {4, 32}}};
const Meshes kH3LightMeshes = {
    {"galaxy", {4, 8}}, {"bh-galaxy", {4, 8}}, {"galaxy_quad", {4, 32}}};

ModelSpec video(std::string log, Meshes meshes, std::string checkpoint) {
  ModelSpec s;
  s.logName = std::move(log);
  s.meshes = std::move(meshes);
  s.checkpoint = std::move(checkpoint);
  return s;
}

ModelSpec image(std::string log, Meshes meshes, std::string checkpoint,
                bool fromConfig = true) {
  ModelSpec s = video(std::move(log), std::move(meshes), std::move(checkpoint));
  s.kind = OutputKind::Image;
  s.checkpointFromConfig = fromConfig;
  return s;
}

DeviceParams imageDevice(long traceRegion, bool l1Small = true) {
  DeviceParams p;
  p.traceRegionBytes = traceRegion;
  if (l1Small) p.l1SmallBytes = 32768;
  return p;
}

template <class T>
std::unique_ptr<DiTModel> make(const RunnerConfig& c, ModelSpec spec) {
  return std::make_unique<T>(c, std::move(spec));
}

struct Entry {
  const char* name;
  std::unique_ptr<DiTModel> (*create)(const RunnerConfig&);
};

const Entry kEntries[] = {
    // Wan2.2
    {"tt-wan2.2",
     [](const RunnerConfig& c) {
       auto s = video("Wan22", kWanMeshes, kWanT2V);
       s.untracedDevices = {"t3k"};  // traced output is noise on a T3K
       return make<WanModel>(c, s);
     }},
    {"tt-wan2.2-i2v",
     [](const RunnerConfig& c) {
       auto s = video("Wan22-I2V", withoutN150(kWanMeshes), kWanI2V);
       s.untracedDevices = {"t3k"};  // same pipeline, same mesh as T2V
       return make<WanI2VModel>(c, s);
     }},
    {"tt-wan2.2-i2v-anisora",
     [](const RunnerConfig& c) {
       auto s = video("Wan22-I2V-AniSora", kBhGalaxy, kWanI2V);
       s.warmupPrompt = "An anime girl smiling, soft lighting";
       return make<WanAniSoraModel>(c, s);
     }},
    {"tt-wan2.2-i2v-distill",
     [](const RunnerConfig& c) {
       return make<WanDistillModel>(
           c, video("Wan22-I2V-Distill", kBhGalaxy, kWanI2V));
     }},
    {"tt-wan2.2-i2v-lora",
     [](const RunnerConfig& c) {
       auto s = video("Wan22-I2V-LoRA", kBhGalaxy, kWanI2V);
       s.warmupPrompt = "A golden retriever running on a sandy beach";
       return make<WanLoraModel>(c, s);
     }},
    {"tt-wan2.2-i2v-lightning",
     [](const RunnerConfig& c) {
       return make<WanLightningModel>(
           c, video("Wan22-I2V-Lightning", kBhGalaxy, kWanI2V));
     }},
    {"tt-wan2.2-t2v-prodia",
     [](const RunnerConfig& c) {
       return make<WanProdiaT2VModel>(
           c, video("Wan22-T2V-Prodia", kProdia, kWanT2V));
     }},
    {"tt-wan2.2-i2v-prodia",
     [](const RunnerConfig& c) {
       return make<WanProdiaI2VModel>(
           c, video("Wan22-I2V-Prodia", kProdia, kWanI2V));
     }},
    // Mochi, LTX
    {"tt-mochi-1",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<MochiModel>(c);
     }},
    {"tt-ltx-2.3-distilled",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<LtxModel>(c);
     }},
    // MiniMax-H3
    {"tt-minimax-h3-t2va",
     [](const RunnerConfig& c) {
       return make<MiniMaxModel>(c, video("MiniMaxH3", kH3Meshes, kH3));
     }},
    {"tt-minimax-h3-fasth3",
     [](const RunnerConfig& c) {
       return make<MiniMaxFastH3Model>(
           c, video("MiniMaxH3-FastH3", kH3Meshes, kH3));
     }},
    {"tt-minimax-h3-fl2va",
     [](const RunnerConfig& c) {
       return make<MiniMaxFL2VAModel>(c,
                                      video("MiniMaxH3-FL2VA", kH3Meshes, kH3));
     }},
    {"tt-minimax-h3-ref2va",
     [](const RunnerConfig& c) {
       return make<MiniMaxRef2VAModel>(
           c, video("MiniMaxH3-Ref2VA", kH3Meshes, kH3));
     }},
    {"tt-minimax-h3-fl2va-lightx2v",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<LightX2VModel<MiniMaxFL2VAModel>>(
           c, video("MiniMaxH3-FL2VA-LightX2V", kH3LightMeshes, kH3), 6.0, 3.0);
     }},
    {"tt-minimax-h3-ref2va-lightx2v",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<LightX2VModel<MiniMaxRef2VAModel>>(
           c, video("MiniMaxH3-Ref2VA-LightX2V", kH3LightMeshes, kH3), 12.0,
           3.0);
     }},
    // Image models
    {"tt-sd3.5",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<ImageModel>(
           c,
           image("SD35", {{"t3k", {2, 4}}, {"galaxy", {4, 8}}},
                 "stabilityai/stable-diffusion-3.5-large", false),
           imageDevice(25'000'000), "sd35");
     }},
    {"tt-flux.1-dev",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<ImageModel>(
           c,
           image("FLUX.1-dev",
                 {{"t3k", {2, 4}},
                  {"galaxy", {4, 8}},
                  {"p150x4", {2, 2}},
                  {"p150x8", {2, 4}},
                  {"p300", {1, 2}},
                  {"p300x2", {2, 2}}},
                 "black-forest-labs/FLUX.1-dev"),
           imageDevice(settingsTraceRegion(c.device, c.modelRunner)), "flux1");
     }},
    {"tt-flux.1-schnell",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<ImageModel>(
           c,
           image("FLUX.1-schnell",
                 {{"t3k", {2, 4}},
                  {"galaxy", {4, 8}},
                  {"p150x4", {2, 2}},
                  {"p150x8", {2, 4}},
                  {"p300", {1, 2}},
                  {"p300x2", {2, 2}}},
                 "black-forest-labs/FLUX.1-schnell"),
           imageDevice(settingsTraceRegion(c.device, c.modelRunner)), "flux1");
     }},
    {"tt-flux.1-kontext-dev",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       auto s = image("FLUX.1-Kontext-dev",
                      {{"p300", {1, 2}}, {"p300x2", {2, 2}}, {"t3k", {2, 4}}},
                      "black-forest-labs/FLUX.1-Kontext-dev");
       s.defaultTraced = false;  // validated untraced
       return std::make_unique<FluxKontextModel>(
           c, s, imageDevice(settingsTraceRegion(c.device, c.modelRunner)));
     }},
    {"tt-motif-image-6b-preview",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<ImageModel>(
           c,
           image("Motif-Image-6B-Preview",
                 {{"t3k", {2, 4}},
                  {"galaxy", {4, 8}},
                  {"p150x8", {2, 4}},
                  {"p300x2", {2, 2}}},
                 "Motif-Technologies/Motif-Image-6B-Preview", false),
           imageDevice(31'000'000), "motif");
     }},
    {"tt-qwen-image",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<ImageModel>(
           c,
           image("Qwen-Image", {{"t3k", {2, 4}}, {"galaxy", {4, 8}}},
                 "Qwen/Qwen-Image"),
           imageDevice(47'000'000, false), "qwen_image");
     }},
    {"tt-qwen-image-2512",
     [](const RunnerConfig& c) -> std::unique_ptr<DiTModel> {
       return std::make_unique<ImageModel>(
           c,
           image("Qwen-Image-2512", {{"t3k", {2, 4}}, {"galaxy", {4, 8}}},
                 "Qwen/Qwen-Image-2512"),
           imageDevice(47'000'000, false), "qwen_image");
     }},
};

}  // namespace

std::vector<std::string> supportedRunners() {
  std::vector<std::string> names;
  for (const auto& e : kEntries) names.push_back(e.name);
  std::sort(names.begin(), names.end());
  return names;
}

std::unique_ptr<DiTModel> createModel(const RunnerConfig& config) {
  for (const auto& e : kEntries) {
    if (config.modelRunner == e.name) return e.create(config);
  }
  std::string known;
  for (const auto& name : supportedRunners()) {
    known += (known.empty() ? "" : ", ") + name;
  }
  throw std::invalid_argument("MODEL_RUNNER='" + config.modelRunner +
                              "' has no model; supported: " + known);
}

}  // namespace tt::dit
