// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/pytypes.h>

#include <chrono>
#include <memory>

#include "runtime/dit/dit_model.hpp"
#include "runtime/dit/dit_runner.hpp"

namespace tt::dit {

/**
 * Runs a DiTModel on the device. C++ opens the mesh through ttnn, builds each
 * call from the model, validates, exports the mp4 and handles errors; the only
 * Python it runs is the model's adapter (python/cpp_dit_adapters), which holds
 * the calls into tt_dit: create the pipeline, call it. All methods need the
 * Python GIL.
 */
class DiTRunner : public IDiTRunner {
 public:
  explicit DiTRunner(std::unique_ptr<DiTModel> model);
  ~DiTRunner() override;

  void openDevice() override;
  void warmup() override;
  void validate(const DiTRequest& request) override;
  DiTResult generate(const DiTRequest& request) override;
  void infer(const DiTRequest& request) override;
  void close() override;

  const DiTModel& model() const { return *model_; }
  /** True when the open mesh spans several MPI ranks. */
  bool meshIsDistributed() const;

 private:
  struct Python;
  /** validate() + the adapter's call; returns its result dict. */
  pybind11::dict callAdapter(const DiTRequest& request);
  /** Turns the adapter's result into the job's output (mp4, images, path). */
  DiTResult exportResult(const DiTRequest& request, pybind11::dict result,
                         std::chrono::steady_clock::time_point started);

  std::unique_ptr<DiTModel> model_;
  std::unique_ptr<Python> python_;
};

/** The runner for config.modelRunner with its mesh open, wrapped in
 *  LockstepRunner when the mesh spans several ranks. Needs the GIL and
 *  prepareVideoPythonPath() done. */
std::unique_ptr<IDiTRunner> createOpenRunner(const RunnerConfig& config);

/** A follower rank's runner, from the environment tt-run gave it (MODEL_RUNNER,
 *  DEVICE, MODEL_WEIGHTS_PATH, TT_VIDEO_OUTPUT_DIR), and its Comm. */
int runFollowerRank(int rank);

}  // namespace tt::dit
