// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// MODEL_RUNNER -> model. One row per runner of the Python server's
// tt_model_runners/dit_runners.py that this server runs: Wan2.2 T2V and I2V.

#include <algorithm>

#include "runtime/dit/device_params.hpp"
#include "runtime/dit/dit_models.hpp"

namespace tt::dit {

namespace {

using Meshes = std::map<std::string, MeshShape>;

const char* const WAN_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";
const char* const WAN_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";

const Meshes WAN_MESHES = {{"n150", {1, 1}},   {"t3k", {2, 4}},
                           {"galaxy", {4, 8}}, {"bh-galaxy", {4, 8}},
                           {"p150x4", {1, 4}}, {"p150x8", {2, 4}},
                           {"p300x2", {2, 2}}, {"galaxy_quad", {4, 32}}};
Meshes withoutN150(Meshes m) {
  m.erase("n150");
  return m;
}

ModelSpec video(std::string log, Meshes meshes, std::string checkpoint) {
  ModelSpec s;
  s.logName = std::move(log);
  s.meshes = std::move(meshes);
  s.checkpoint = std::move(checkpoint);
  return s;
}

template <class T>
std::unique_ptr<DiTModel> make(const RunnerConfig& c, ModelSpec spec) {
  return std::make_unique<T>(c, std::move(spec));
}

struct Entry {
  const char* name;
  std::unique_ptr<DiTModel> (*create)(const RunnerConfig&);
};

const Entry ENTRIES[] = {
    // Wan2.2
    {"tt-wan2.2",
     [](const RunnerConfig& c) {
       auto s = video("Wan22", WAN_MESHES, WAN_T2V);
       s.untracedDevices = {"t3k"};  // traced output is noise on a T3K
       return make<WanModel>(c, s);
     }},
    {"tt-wan2.2-i2v",
     [](const RunnerConfig& c) {
       auto s = video("Wan22-I2V", withoutN150(WAN_MESHES), WAN_I2V);
       s.untracedDevices = {"t3k"};  // same pipeline, same mesh as T2V
       return make<WanI2VModel>(c, s);
     }},
};

}  // namespace

std::vector<std::string> supportedRunners() {
  std::vector<std::string> names;
  for (const auto& e : ENTRIES) names.push_back(e.name);
  std::sort(names.begin(), names.end());
  return names;
}

std::unique_ptr<DiTModel> createModel(const RunnerConfig& config) {
  for (const auto& e : ENTRIES) {
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
