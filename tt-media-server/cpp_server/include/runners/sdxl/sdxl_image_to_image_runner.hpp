// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runners/sdxl/sdxl_base_runner.hpp"

namespace tt::runners::sdxl {

class SDXLImageToImageRunner : public SDXLBaseRunner {
 public:
  explicit SDXLImageToImageRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLImageToImageRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;
  void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& request) override;
  bool areBatchCompatible(
      const domain::ImageGenerateRequest& a,
      const domain::ImageGenerateRequest& b) const override;

  /** base64 -> PIL -> torch tensor (1, C, H, W). */
  py::object preprocessImage(const std::string& base64Image) const;

  /** Stack per-request preprocessed images; pad trailing rows with copies of
   *  the first slot. The pipeline runs over padded rows; outputs are
   *  discarded in `postProcessImages`. */
  py::object stackImageBatch(
      const std::vector<domain::ImageGenerateRequest>& requests) const;
};

}  // namespace tt::runners::sdxl
