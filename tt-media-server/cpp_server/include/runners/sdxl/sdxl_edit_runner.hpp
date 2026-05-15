// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runners/sdxl/sdxl_image_to_image_runner.hpp"

namespace tt::runners::sdxl {

class SDXLEditRunner : public SDXLImageToImageRunner {
 public:
  explicit SDXLEditRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLEditRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;

 private:
  py::object preprocessMask(const std::string& base64Mask) const;
  py::object stackMaskBatch(
      const std::vector<domain::ImageGenerateRequest>& requests) const;
};

}  // namespace tt::runners::sdxl
