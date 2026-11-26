// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runtime/dit/dit_model.hpp"

/**
 * The model classes. Values (mesh shapes, fabric, trace regions, guidance,
 * step counts, resolutions) mirror the Python server's
 * tt_model_runners/dit_runners.py and config/constants.py; a unit test
 * compares the runner list and mesh shapes with that config. Keep the rest in
 * sync by hand.
 */
namespace tt::dit {

// -- Wan2.2 -----------------------------------------------------------------

constexpr int kWan22NumFrames = 81;

/** (height, width): 720p on Galaxy-class meshes, 480p on smaller ones. */
std::pair<int, int> wan22Resolution(MeshShape mesh);

/** Wan2.2 text-to-video. */
class WanModel : public DiTModel {
 public:
  WanModel(RunnerConfig config, ModelSpec spec);
  std::string adapterName() const override { return "wan_t2v"; }
  Json::Value adapterOptions(const HostEnv& host) const override;
  Json::Value buildCall(const DiTRequest& request) const override;
  DeviceParams deviceParams(const HostEnv& host) const override;

 protected:
  int height_;
  int width_;
};

/** Image-conditioned Wan2.2: image_prompts go to the pipeline. */
class WanI2VModel : public WanModel {
 public:
  WanI2VModel(RunnerConfig config, ModelSpec spec)
      : WanModel(std::move(config), std::move(spec)) {}
  std::string adapterName() const override { return "wan_i2v"; }
  Json::Value buildCall(const DiTRequest& request) const override;
  bool requiresImageConditioning() const override { return true; }
};

}  // namespace tt::dit
