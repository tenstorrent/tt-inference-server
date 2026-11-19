// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/dit_model.hpp"

#include <filesystem>
#include <system_error>

#include "runtime/dit/dit_env.hpp"

namespace tt::dit {

namespace {

const std::pair<const char*, MeshShape> kMeshEnvOverrides[] = {
    {"SD_3_5_FAST", {4, 8}},
    {"SD_3_5_BASE", {2, 4}},
    {"TP2", {2, 1}},
    {"SP_MESH_4X32", {4, 32}},
};

}  // namespace

MeshShape resolveMesh(const std::map<std::string, MeshShape>& meshes,
                      const std::string& runner, const std::string& device) {
  for (const auto& [name, shape] : kMeshEnvOverrides) {
    const auto value = envString(name);
    if (value && lowered(*value) == "true") return shape;
  }
  const auto it = meshes.find(lowered(trimmed(device)));
  if (it == meshes.end()) {
    std::string known;
    for (const auto& [name, shape] : meshes) {
      known += (known.empty() ? "" : ", ") + name;
    }
    throw std::invalid_argument("DEVICE='" + device +
                                "' is not supported for " + runner +
                                "; expected one of: " + known);
  }
  return it->second;
}

std::string resolveCheckpoint(const std::string& modelWeightsPath,
                              const std::string& defaultCheckpoint) {
  namespace fs = std::filesystem;
  if (const auto dir = envString("MODEL_WEIGHTS_DIR"); dir && !dir->empty()) {
    std::error_code ec;
    if (fs::is_directory(*dir, ec) &&
        fs::directory_iterator(*dir, ec) != fs::directory_iterator()) {
      return *dir;
    }
  }
  return modelWeightsPath.empty() ? defaultCheckpoint : modelWeightsPath;
}

DiTModel::DiTModel(RunnerConfig config, ModelSpec spec)
    : config_(std::move(config)),
      spec_(std::move(spec)),
      mesh_(resolveMesh(spec_.meshes, config_.modelRunner, config_.device)),
      checkpoint_(
          spec_.checkpointFromConfig
              ? resolveCheckpoint(config_.modelWeightsPath, spec_.checkpoint)
              : spec_.checkpoint),
      traced_(envFlag("TT_VIDEO_TRACED",
                      spec_.defaultTraced &&
                          spec_.untracedDevices.count(config_.device) == 0)) {}

void DiTModel::validate(const DiTRequest& request) const {
  if (requiresImageConditioning()) validateImagePrompts(request.imagePrompts);
}

DiTRequest DiTModel::warmupRequest() const {
  DiTRequest r;
  r.prompt = spec_.warmupPrompt;
  r.negativePrompt = "";
  r.numInferenceSteps = spec_.warmupSteps;
  if (requiresImageConditioning()) {
    // A small black PNG: the pipelines resize the image, so its size does not
    // matter, and it exercises the same kernels as a photo.
    static const char* kBlackPng =
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAIklEQVR4nO3BAQ0A"
        "AADCoPdPbQ8HFAAAAAAAAAAAAAAA8G4wQAABiwCo9wAAAABJRU5ErkJggg==";
    r.imagePrompts.push_back({kBlackPng, 0});
  }
  return r;
}

int DiTModel::stepsFor(const DiTRequest& request) const {
  return request.numInferenceSteps.value_or(0) > 0 ? *request.numInferenceSteps
                                                   : spec_.defaultSteps;
}

int DiTModel::steps(const DiTRequest& request) const {
  return stepsFor(request);
}

int DiTModel::seed(const DiTRequest& request) {
  return request.seed.value_or(0);
}

double DiTModel::fps(const DiTRequest& request) const {
  return request.fps.value_or(0) > 0 ? *request.fps : spec_.defaultFps;
}

}  // namespace tt::dit
