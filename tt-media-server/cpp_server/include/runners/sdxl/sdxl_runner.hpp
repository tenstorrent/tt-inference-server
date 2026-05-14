// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/embed.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "runners/media_runner.hpp"

namespace tt::runners::sdxl {

namespace py = pybind11;

/**
 * C++ port of `tt_model_runners.base_sdxl_runner.BaseSDXLRunner`. Drives the
 * tt-metal SDXL pipeline via pybind11; the embedded interpreter is reused
 * across requests and all Python access is guarded by the GIL.
 */
class SDXLBaseRunner : public IMediaRunner<domain::ImageGenerateRequest,
                                           std::vector<std::string>> {
 public:
  ~SDXLBaseRunner() override;

  SDXLBaseRunner(const SDXLBaseRunner&) = delete;
  SDXLBaseRunner& operator=(const SDXLBaseRunner&) = delete;

  bool warmup() override;
  std::vector<std::string> run(
      const domain::ImageGenerateRequest& request) override;
  void stop() override;

 protected:
  explicit SDXLBaseRunner(const config::ImageConfig& config);

  virtual py::object loadDiffusersPipeline() = 0;
  virtual void distributeBlock() = 0;
  virtual void prepareInputTensorsForIteration(py::object tensors) = 0;
  virtual py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) = 0;
  virtual domain::ImageGenerateRequest warmupRequest() const = 0;
  virtual void applyModeSpecificSettings(
      const domain::ImageGenerateRequest& /*request*/) {}

  /** Port of `BaseSDXLRunner.is_request_batchable`. Override in subclasses to
   *  add mode-specific fields (e.g. `strength` for img2img). */
  virtual bool areBatchCompatible(const domain::ImageGenerateRequest& a,
                                  const domain::ImageGenerateRequest& b) const;

  /** Run `work` with a hard timeout. Caller MUST NOT hold the GIL. */
  static void runWithTimeout(const std::string& tag, unsigned timeoutSeconds,
                             const std::function<void()>& work);

  /** Encode the first `requests.size()` entries of `imgsList`; trailing
   *  entries are dummy padding produced by `processPrompts`. Each request
   *  selects its own format / quality. */
  std::vector<std::string> postProcessImages(
      const py::object& imgsList,
      const std::vector<domain::ImageGenerateRequest>& requests) const;

  struct PromptPack {
    std::vector<std::string> prompts;
    std::optional<std::vector<std::string>> negative_prompts;
    std::optional<std::vector<std::string>> prompts_2;
    std::optional<std::string> negative_prompt_2;
    int needed_padding = 0;
  };
  PromptPack processPrompts(
      const std::vector<domain::ImageGenerateRequest>& requests) const;

  void injectLoraTriggers(std::vector<std::string>& prompts,
                          const std::optional<std::string>& loraPath) const;
  void applyRequestSettings(const domain::ImageGenerateRequest& request);
  void ensureLoraState(const domain::ImageGenerateRequest& request);
  void initDevice();
  void runFullWarmup();
  py::dict pipelineDeviceParams();

  config::ImageConfig config_;
  size_t batch_size_ = 1;
  bool is_tensor_parallel_ = false;
  bool initialized_ = false;

  py::object ttnn_device_;
  py::object pipeline_;
  py::object tt_sdxl_;
  py::object torch_module_;
  py::object ttnn_module_;

  std::optional<std::string> current_lora_path_;
  std::optional<float> current_lora_scale_;

 private:
  struct BatchSlot {
    domain::ImageGenerateRequest request;
    std::promise<std::vector<std::string>> promise;
  };

  void batcherLoop();
  void runBatch(std::vector<BatchSlot>& batch);

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<BatchSlot> queue_;
  std::atomic<bool> batcher_stop_{false};
  std::thread batcher_thread_;
};

class SDXLGenerateRunner : public SDXLBaseRunner {
 public:
  explicit SDXLGenerateRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLGenerateRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;
};

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
