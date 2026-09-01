// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "runtime/runners/embedding_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <stdexcept>

#include "config/types.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace tt::runners {

namespace {

// Take the first line of a Python error (the "ValueError: ..." part); the
// full traceback is logged separately and is too long for a response field.
std::string firstLine(const std::string& s) {
  const auto pos = s.find('\n');
  return pos == std::string::npos ? s : s.substr(0, pos);
}

// Prepend TT_METAL_HOME to sys.path if not already there. models.demos is a
// plain directory tree inside the tt-metal checkout, so the interpreter can
// only import the generator classes when that root is on the path. Deliberate
// pinning: resolve from the configured location, not from whatever the
// launching shell happened to have in PYTHONPATH.
void ensureSysPath() {
  const char* metalHome = std::getenv("TT_METAL_HOME");
  if (!metalHome || !*metalHome) return;
  py::list sysPath = py::module_::import("sys").attr("path");
  for (const auto& entry : sysPath) {
    if (py::str(entry).cast<std::string>() == metalHome) return;
  }
  sysPath.attr("insert")(0, py::str(metalHome));
  TT_LOG_INFO("[EmbeddingRunner] Prepended TT_METAL_HOME to sys.path: {}",
              metalHome);
}

// Mirror utils/torch_utils.set_torch_thread_limits, which the Python worker
// called before touching the model. TORCH_NUM_THREADS is exported by
// embedding_service.cpp alongside OMP/MKL_NUM_THREADS; the env vars alone
// only size torch's intra-op pool, the interop pool must be capped through
// the API. Both setters throw once their pool has started, hence the get()
// guards and the placement before anything else imports torch.
void applyTorchThreadLimits() {
  int numThreads = 1;
  if (const char* env = std::getenv("TORCH_NUM_THREADS"); env && *env) {
    const int parsed = std::atoi(env);
    if (parsed > 0) numThreads = parsed;
  }
  py::module_ torch = py::module_::import("torch");
  if (torch.attr("get_num_threads")().cast<int>() != numThreads) {
    torch.attr("set_num_threads")(numThreads);
  }
  if (torch.attr("get_num_interop_threads")().cast<int>() != numThreads) {
    torch.attr("set_num_interop_threads")(numThreads);
  }
  TT_LOG_INFO("[EmbeddingRunner] torch thread limits set to {}", numThreads);
}

}  // namespace

namespace detail {

/**
 * Template-method base: owns the full pipeline (device open, tokenizer,
 * model construction, warmup forward, tokenize->forward->extract, close) and
 * defers the model-specific steps to virtuals. A new model is onboarded by
 * adding a subclass below plus a catalog row in settings.cpp.
 */
struct EmbeddingImpl {
  config::EmbeddingConfig config;
  py::object ttnn;       // the imported ttnn module
  py::object device;     // ttnn mesh device
  py::object tokenizer;  // transformers.AutoTokenizer instance
  py::object model;      // the tt-metal generator class instance

  explicit EmbeddingImpl(const config::EmbeddingConfig& cfg) : config(cfg) {}

  virtual ~EmbeddingImpl() { release(); }

  // ---- virtual steps (per model) -----------------------------------------

  /** Import path of the tt-metal module holding the generator class. */
  virtual const char* modelModule() const = 0;

  /** Generator class name inside modelModule(). */
  virtual const char* modelClass() const = 0;

  /** Add the constructor kwargs that differ per model (dtypes, location
   * generator). The shared device/max_batch_size/max_seq_len/model_name
   * kwargs are already set when this is called; the GIL is held and `ttnn`
   * is imported. */
  virtual void addModelKwargs(py::dict& kwargs) const = 0;

  /** Pull the dense embedding tensor out of forward()'s result. Default:
   * forward() returns the tensor itself. */
  virtual py::object extractDense(const py::object& result) const {
    return result;
  }

  // ---- template methods (shared) ------------------------------------------

  bool initialize() {
    // Boot the interpreter once per process. pybind11 leaves the GIL held
    // after initialization; release it at the end of warmup-time setup so
    // every later entry point can acquire it.
    const bool ownsInterpreter = !Py_IsInitialized();
    if (ownsInterpreter) {
      py::initialize_interpreter();
      TT_LOG_INFO("[EmbeddingRunner] Python interpreter initialized");
    }

    bool ok = false;
    {
      py::gil_scoped_acquire gil;
      try {
        ensureSysPath();
        applyTorchThreadLimits();

        ttnn = py::module_::import("ttnn");
        openMeshDevice();

        tokenizer = py::module_::import("transformers")
                        .attr("AutoTokenizer")
                        .attr("from_pretrained")(config.hf_model_id);
        TT_LOG_INFO("[EmbeddingRunner] Tokenizer loaded for {}",
                    config.hf_model_id);

        // The Python runner exported HF_MODEL before loading; tt-metal's
        // tt_transformers config resolves the checkpoint from it (the Qwen
        // path asserts without it). This must go through os.environ, not C
        // setenv(): Python snapshots the environment when the interpreter
        // starts, so C-level writes made after that are invisible to
        // os.getenv.
        py::module_::import("os").attr("environ")[py::str("HF_MODEL")] =
            py::str(config.hf_model_id);

        py::object cls = py::module_::import(modelModule()).attr(modelClass());
        py::dict kwargs;
        kwargs["device"] = device;
        kwargs["max_batch_size"] = config.max_batch_size;
        kwargs["max_seq_len"] = config.max_seq_len;
        kwargs["model_name"] = config.hf_model_id;
        addModelKwargs(kwargs);
        model = cls(**kwargs);
        TT_LOG_INFO("[EmbeddingRunner] {}.{} constructed", modelModule(),
                    modelClass());

        // One real forward pass so weight upload, tracing, and kernel
        // compilation happen now rather than on the first request.
        py::object warm =
            tokenize(py::make_tuple("The capital of France is "
                                    "Paris"));
        forwardAndSync(warm);
        ok = true;
        TT_LOG_INFO("[EmbeddingRunner] Warmup forward pass completed");
      } catch (const py::error_already_set& e) {
        TT_LOG_ERROR("[EmbeddingRunner] Warmup failed:\n{}", e.what());
        ok = false;
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[EmbeddingRunner] Warmup failed: {}", e.what());
        ok = false;
      }
    }

    if (ownsInterpreter) {
      PyEval_SaveThread();
    }
    return ok;
  }

  std::vector<domain::EmbeddingResponse> runInference(
      const std::vector<domain::EmbeddingRequest>& requests) {
    std::vector<domain::EmbeddingResponse> responses;
    responses.reserve(requests.size());

    py::gil_scoped_acquire gil;
    try {
      // Same contract as the Python runner's _validate_requests: a model
      // mismatch anywhere fails the whole batch.
      for (const auto& req : requests) {
        if (req.model != config.hf_model_id) {
          const std::string message =
              "Only " + config.hf_model_id + " embeddings are supported";
          for (const auto& r : requests) {
            domain::EmbeddingResponse resp(r.task_id);
            resp.error = message;
            responses.push_back(std::move(resp));
          }
          return responses;
        }
      }

      py::list texts;
      for (const auto& req : requests) {
        texts.append(req.input);
      }

      py::object tokenized = tokenize(texts);
      py::object result = forwardAndSync(tokenized);
      py::object dense = extractDense(result);

      py::object attentionMask = tokenized.attr("get")("attention_mask");
      std::vector<int> tokenCounts;
      if (!attentionMask.is_none()) {
        tokenCounts = attentionMask.attr("sum")("dim"_a = 1)
                          .attr("tolist")()
                          .cast<std::vector<int>>();
      }

      // Row i of the (possibly batch-padded) result answers requests[i];
      // the pairing is positional by contract.
      for (size_t i = 0; i < requests.size(); ++i) {
        domain::EmbeddingResponse resp(requests[i].task_id);
        resp.model = requests[i].model;
        resp.embedding = dense[py::int_(i)]
                             .attr("cpu")()
                             .attr("numpy")()
                             .attr("tolist")()
                             .cast<std::vector<float>>();
        resp.total_tokens = i < tokenCounts.size() ? tokenCounts[i] : 0;
        responses.push_back(std::move(resp));
      }

      TT_LOG_DEBUG("[EmbeddingRunner] Processed {} embedding requests",
                   responses.size());
    } catch (const py::error_already_set& e) {
      // Surface the real Python error to every caller in the batch instead
      // of a generic "no response". The full traceback goes to the log.
      TT_LOG_ERROR("[EmbeddingRunner] Inference failed:\n{}", e.what());
      const std::string message = firstLine(e.what());
      responses.clear();
      for (const auto& req : requests) {
        domain::EmbeddingResponse resp(req.task_id);
        resp.error = message;
        responses.push_back(std::move(resp));
      }
    }

    return responses;
  }

  void release() {
    if (!Py_IsInitialized()) return;
    py::gil_scoped_acquire gil;
    if (ttnn && device && !device.is_none()) {
      try {
        ttnn.attr("close_mesh_device")(device);
        TT_LOG_INFO("[EmbeddingRunner] Mesh device closed");
      } catch (const py::error_already_set& e) {
        TT_LOG_WARN("[EmbeddingRunner] close_mesh_device failed: {}",
                    firstLine(e.what()));
      }
    }
    model = py::object();
    tokenizer = py::object();
    device = py::object();
    ttnn = py::object();
    // The interpreter itself is never finalized: other components (and a
    // possible restart of this runner) may still need it.
  }

 protected:
  /** The identity model_location_generator the Python runners passed: the
   * models resolve weights straight from the HuggingFace id. */
  py::object identityLocationGenerator() const {
    return py::cpp_function([](py::object version) { return version; });
  }

 private:
  // Mirrors BaseMetalDeviceRunner._mesh_device for the embedding models:
  // none of them set dispatch knobs, so DispatchCoreConfig gets all-None
  // (defaults), and fabric config is never used on this path.
  void openMeshDevice() {
    py::object meshShape = ttnn.attr("MeshShape")(py::cast(config.mesh_shape));
    py::dict params;
    params["dispatch_core_config"] =
        ttnn.attr("DispatchCoreConfig")(py::none(), py::none(), py::none());
    if (config.trace_region_size > 0) {
      params["trace_region_size"] = config.trace_region_size;
    }
    if (config.num_command_queues > 0) {
      params["num_command_queues"] = config.num_command_queues;
    }
    device =
        ttnn.attr("open_mesh_device")("mesh_shape"_a = meshShape, **params);
    TT_LOG_INFO("[EmbeddingRunner] Opened mesh device with {} device(s)",
                device.attr("get_num_devices")().cast<size_t>());
  }

  // Same call the Python EmbeddingTokenizer made, so token streams are
  // byte-identical to the Python server's.
  py::object tokenize(const py::object& texts) const {
    return tokenizer(texts, "padding"_a = true, "truncation"_a = true,
                     "max_length"_a = config.max_seq_len,
                     "return_tensors"_a = "pt");
  }

  py::object forwardAndSync(const py::object& tokenized) {
    py::object result = model.attr("forward")(
        tokenized[py::str("input_ids")],
        "attention_mask"_a = tokenized.attr("get")("attention_mask"));
    ttnn.attr("synchronize_device")(device);
    return result;
  }
};

}  // namespace detail

namespace {

struct BgeLargeEnImpl final : detail::EmbeddingImpl {
  using EmbeddingImpl::EmbeddingImpl;

  const char* modelModule() const override {
    return "models.demos.wormhole.bge_large_en.demo.generator_vllm";
  }
  const char* modelClass() const override { return "BGEForEmbedding"; }

  void addModelKwargs(py::dict& kwargs) const override {
    kwargs["model_location_generator"] = identityLocationGenerator();
    kwargs["act_dtype"] = ttnn.attr("bfloat16");
    kwargs["weight_dtype"] = ttnn.attr("bfloat8_b");
  }
};

struct BgeM3Impl final : detail::EmbeddingImpl {
  using EmbeddingImpl::EmbeddingImpl;

  const char* modelModule() const override {
    return "models.demos.wormhole.bge_m3.demo.generator_vllm";
  }
  const char* modelClass() const override { return "BgeM3ForEmbedding"; }

  void addModelKwargs(py::dict& kwargs) const override {
    kwargs["dtype"] = ttnn.attr("bfloat8_b");
  }

  // BGE-M3's forward returns {"dense_vecs": ..., ...} because the model can
  // also produce sparse/colbert outputs; we serve dense only.
  py::object extractDense(const py::object& result) const override {
    return result[py::str("dense_vecs")];
  }
};

struct Qwen3Embedding8bImpl final : detail::EmbeddingImpl {
  using EmbeddingImpl::EmbeddingImpl;

  const char* modelModule() const override {
    return "models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm";
  }
  const char* modelClass() const override { return "Qwen3ForEmbedding"; }

  void addModelKwargs(py::dict& kwargs) const override {
    kwargs["model_location_generator"] = identityLocationGenerator();
    kwargs["act_dtype"] = ttnn.attr("bfloat16");
    kwargs["weight_dtype"] = ttnn.attr("bfloat8_b");
  }
};

std::unique_ptr<detail::EmbeddingImpl> makeImpl(
    const config::EmbeddingConfig& cfg) {
  switch (cfg.runner_type) {
    case config::ModelRunnerType::TT_BGE_LARGE_EN:
      return std::make_unique<BgeLargeEnImpl>(cfg);
    case config::ModelRunnerType::TT_BGE_M3:
      return std::make_unique<BgeM3Impl>(cfg);
    case config::ModelRunnerType::TT_QWEN_EMBEDDING_8B:
      return std::make_unique<Qwen3Embedding8bImpl>(cfg);
    default:
      throw std::runtime_error(
          "[EmbeddingRunner] runner_type=" + config::toString(cfg.runner_type) +
          " is not a tt-metal embedding model");
  }
}

}  // namespace

// Public interface implementation

EmbeddingRunner::EmbeddingRunner(const config::EmbeddingConfig& config)
    : config_(config), impl_(makeImpl(config)) {
  TT_LOG_INFO("[EmbeddingRunner] Created for model {} on device {} (worker {})",
              config_.hf_model_id, config_.visible_devices, config_.worker_id);
}

EmbeddingRunner::~EmbeddingRunner() { close(); }

bool EmbeddingRunner::warmup() {
  TT_LOG_INFO("[EmbeddingRunner] Starting warmup for {} on device {}",
              config_.hf_model_id, config_.visible_devices);

  if (!impl_->initialize()) {
    return false;
  }

  TT_LOG_INFO("[EmbeddingRunner] Warmup complete for device {}",
              config_.visible_devices);
  return true;
}

void EmbeddingRunner::close() {
  if (impl_) {
    impl_->release();
  }
}

std::vector<domain::EmbeddingResponse> EmbeddingRunner::run(
    const std::vector<domain::EmbeddingRequest>& requests) {
  if (!impl_ || !impl_->model) {
    TT_LOG_ERROR("[EmbeddingRunner] Runner not initialized");
    return {};
  }

  return impl_->runInference(requests);
}

}  // namespace tt::runners
