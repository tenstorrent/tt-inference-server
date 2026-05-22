// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/embed.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"
#include "runtime/runners/media_runner.hpp"

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

  static void runAndCheckDuration(const std::string& tag,
                                  unsigned timeoutSeconds,
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

  py::object& ttnn_device();
  py::object& pipeline();
  py::object& tt_sdxl();
  py::object& torch_module();
  py::object& ttnn_module();
  const py::object& ttnn_device() const;
  const py::object& pipeline() const;
  const py::object& tt_sdxl() const;
  const py::object& torch_module() const;
  const py::object& ttnn_module() const;

  std::optional<std::string> current_lora_path_;
  std::optional<float> current_lora_scale_;

 private:
  struct PythonState;
  std::unique_ptr<PythonState> python_;

  struct BatchSlot {
    explicit BatchSlot(const domain::ImageGenerateRequest& req)
        : request(req) {}
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

}  // namespace tt::runners::sdxl
