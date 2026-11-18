// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/dit_python_runner.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "runtime/dit/dit_env.hpp"
#include "runtime/dit/dit_multihost.hpp"
#include "runtime/dit/media_export.hpp"
#include "utils/logger.hpp"

namespace py = pybind11;

namespace tt::dit {

namespace {

constexpr int kCpuThreads = 2;
constexpr int kTorchThreads = 1;
constexpr uint64_t kMessageTag = 0x7E57;
constexpr int kRootRank = 0;

double secondsSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
      .count();
}

py::object toPython(const Json::Value& value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  return py::module_::import("json").attr("loads")(
      Json::writeString(writer, value));
}

/** The adapters call torch and numpy: keep them from oversubscribing the CPU
 *  (the Python server's device worker sets the same limits). */
void setCpuThreadLimits() {
  setEnvIfChanged("OMP_NUM_THREADS", std::to_string(kCpuThreads));
  setEnvIfChanged("MKL_NUM_THREADS", std::to_string(kCpuThreads));
  setEnvIfChanged("TORCH_NUM_THREADS", std::to_string(kTorchThreads));
  py::module_ torch = py::module_::import("torch");
  if (torch.attr("get_num_threads")().cast<int>() != kTorchThreads) {
    torch.attr("set_num_threads")(kTorchThreads);
  }
  if (torch.attr("get_num_interop_threads")().cast<int>() != kTorchThreads) {
    torch.attr("set_num_interop_threads")(kTorchThreads);
  }
}

bool isBlackhole(const py::module_& ttnn) {
  return ttnn.attr("device").attr("is_blackhole")().cast<bool>();
}

/** Set the fabric and open the mesh. Same steps as the Python server's
 *  BaseMetalDeviceRunner._mesh_device. */
py::object openMesh(MeshShape mesh, const DeviceParams& params) {
  py::module_ ttnn = py::module_::import("ttnn");
  const py::list ids = ttnn.attr("get_device_ids")();
  if (py::len(ids) == 0) throw std::runtime_error("no TTNN devices available");
  TT_LOG_INFO("[DiT] Found {} TTNN devices", py::len(ids));

  py::object fabric = ttnn.attr("FabricConfig");
  py::object reliability = ttnn.attr("FabricReliabilityMode");
  py::object router = ttnn.attr("FabricRouterConfig")();
  if (params.routerPayloadBytes) {
    router.attr("max_packet_payload_size_bytes") = *params.routerPayloadBytes;
  }
  ttnn.attr("set_fabric_config")(
      fabric.attr(params.ring ? "FABRIC_1D_RING" : "FABRIC_1D"),
      reliability.attr(params.relaxedInit ? "RELAXED_INIT" : "STRICT_INIT"),
      py::none(), ttnn.attr("FabricTensixConfig").attr("DISABLED"),
      ttnn.attr("FabricUDMMode").attr("DISABLED"),
      ttnn.attr("FabricManagerMode").attr("DEFAULT"), router);

  py::dict kwargs;
  kwargs["dispatch_core_config"] =
      ttnn.attr("DispatchCoreConfig")(py::none(), py::none(), py::none());
  if (params.l1SmallBytes) kwargs["l1_small_size"] = *params.l1SmallBytes;
  if (params.traceRegionBytes) {
    kwargs["trace_region_size"] = *params.traceRegionBytes;
  }
  kwargs["mesh_shape"] =
      ttnn.attr("MeshShape")(py::make_tuple(mesh.rows, mesh.cols));
  try {
    py::object meshDevice = ttnn.attr("open_mesh_device")(**kwargs);
    TT_LOG_INFO("[DiT] Opened mesh device with {} chips",
                meshDevice.attr("get_num_devices")().cast<int>());
    return meshDevice;
  } catch (...) {
    try {
      ttnn.attr("set_fabric_config")(fabric.attr("DISABLED"));
    } catch (const py::error_already_set& e) {
      TT_LOG_WARN("[DiT] fabric reset after a failed open: {}", e.what());
    }
    throw;
  }
}

ArrayView viewOf(const py::array& array) {
  ArrayView v;
  const py::buffer_info info = array.request();
  v.data = info.ptr;
  for (auto d : info.shape) v.shape.push_back(static_cast<size_t>(d));
  if (info.format == py::format_descriptor<uint8_t>::format()) {
    v.dtype = DType::U8;
  } else if (info.format == py::format_descriptor<float>::format()) {
    v.dtype = DType::F32;
  } else if (info.format == py::format_descriptor<double>::format()) {
    v.dtype = DType::F64;
  } else {
    throw std::runtime_error("adapter returned an array of format '" +
                             info.format +
                             "'; expected uint8, float32 or float64");
  }
  return v;
}

/** The adapter must return C-contiguous arrays of those dtypes. */
py::array contiguous(const py::object& obj) {
  py::array a = py::array::ensure(obj, py::array::c_style);
  if (!a)
    throw std::runtime_error("adapter returned a value that is not an array");
  return a;
}

void deleteIfFile(const std::string& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

/** Rank 0 -> every other rank over tt-metal's MPI context. Messages are a
 *  big-endian 8-byte length, then the payload. */
class MpiComm : public Comm {
 public:
  void broadcast(const std::string& message) override {
    py::module_ ttnn = py::module_::import("ttnn");
    const int rank = ttnn.attr("distributed_context_get_rank")().cast<int>();
    if (rank != kRootRank) {
      throw std::runtime_error("broadcast called on rank " +
                               std::to_string(rank));
    }
    std::string header(8, '\0');
    for (int i = 0; i < 8; ++i) {
      header[7 - i] = static_cast<char>((message.size() >> (8 * i)) & 0xFF);
    }
    const int size = ttnn.attr("distributed_context_get_size")().cast<int>();
    for (int dest = kRootRank + 1; dest < size; ++dest) {
      ttnn.attr("distributed_context_send_bytes")(py::bytes(header), dest,
                                                  kMessageTag);
      ttnn.attr("distributed_context_send_bytes")(py::bytes(message), dest,
                                                  kMessageTag);
    }
  }

  std::string receive() override {
    py::module_ ttnn = py::module_::import("ttnn");
    const std::string header =
        ttnn.attr("distributed_context_recv_bytes")(8, kRootRank, kMessageTag)
            .cast<std::string>();
    uint64_t length = 0;
    for (unsigned char c : header) length = (length << 8) | c;
    return ttnn
        .attr("distributed_context_recv_bytes")(length, kRootRank, kMessageTag)
        .cast<std::string>();
  }

  void barrier() override {
    py::module_::import("ttnn").attr("distributed_context_barrier")();
  }
};

}  // namespace

struct __attribute__((visibility("hidden"))) DiTRunner::Python {
  py::object mesh;     // ttnn MeshDevice
  py::object adapter;  // cpp_dit_adapters adapter bound to the pipeline
  HostEnv host;
};

DiTRunner::DiTRunner(std::unique_ptr<DiTModel> model)
    : model_(std::move(model)), python_(std::make_unique<Python>()) {
  TT_LOG_INFO(
      "[DiT] {} runner={} worker={} device={} mesh={}x{} checkpoint={} "
      "traced={}",
      model_->spec().logName, model_->runnerName(), model_->config().workerId,
      model_->config().device, model_->mesh().rows, model_->mesh().cols,
      model_->checkpoint(), model_->traced());
}

DiTRunner::~DiTRunner() = default;

void DiTRunner::openDevice() {
  if (python_->mesh) return;
  setCpuThreadLimits();
  py::module_ ttnn = py::module_::import("ttnn");
  python_->host.blackhole = isBlackhole(ttnn);
  python_->host.minimaxPreset = [](MeshShape mesh) {
    py::dict preset = py::module_::import("cpp_dit_adapters")
                          .attr("minimax_preset")(mesh.rows, mesh.cols);
    return MiniMaxPreset{preset["ring"].cast<bool>(),
                         preset["trace_denoise"].cast<bool>()};
  };
  std::error_code ec;
  std::filesystem::create_directories(model_->config().outputDir, ec);
  python_->mesh = openMesh(model_->mesh(), model_->deviceParams(python_->host));
}

void DiTRunner::warmup() {
  if (!python_->mesh) throw std::runtime_error("openDevice() must come first");
  for (const auto& [name, value] : model_->envDefaults()) {
    ::setenv(name.c_str(), value.c_str(), 0);  // a deployment can pin a flag
  }
  auto started = std::chrono::steady_clock::now();
  py::module_ adapters = py::module_::import("cpp_dit_adapters");
  python_->adapter =
      adapters.attr("create")(model_->adapterName(), python_->mesh,
                              toPython(model_->adapterOptions(python_->host)));
  TT_LOG_INFO("[DiT] {} pipeline created in {:.1f} s", model_->spec().logName,
              secondsSince(started));
  if (!model_->runsWarmupRequest()) return;  // the pipeline warms itself up
  started = std::chrono::steady_clock::now();
  py::dict result = callAdapter(model_->warmupRequest());
  // Whatever the warmup run wrote is not a job's output.
  if (py::str(result["kind"]).cast<std::string>() == "path") {
    deleteIfFile(result["path"].cast<std::string>());
  }
  TT_LOG_INFO("[DiT] {} warmup run done in {:.1f} s", model_->spec().logName,
              secondsSince(started));
}

void DiTRunner::validate(const DiTRequest& request) {
  // Anything the model's checks or the call builder throw about a request is
  // the client's mistake (a number too large for an int, a field of the wrong
  // JSON type, ...), not a failure of the runner.
  try {
    model_->validate(request);
    model_->buildCall(request);
  } catch (const RequestError&) {
    throw;
  } catch (const std::exception& e) {
    throw RequestError(std::string("invalid request: ") + e.what());
  }
}

DiTResult DiTRunner::generate(const DiTRequest& request) {
  py::dict result = callAdapter(request);
  const auto started = std::chrono::steady_clock::now();
  try {
    return exportResult(request, result, started);
  } catch (const ExportError&) {
    throw;
  } catch (const std::exception& e) {
    throw ExportError(std::string("cannot export the result: ") + e.what());
  }
}

DiTResult DiTRunner::exportResult(
    const DiTRequest& request, py::dict result,
    std::chrono::steady_clock::time_point started) {
  DiTResult out;
  out.kind = model_->outputKind();
  const std::string kind = py::str(result["kind"]).cast<std::string>();
  const std::string& dir = model_->config().outputDir;
  if (kind == "path") {
    out.path = result["path"].cast<std::string>();
  } else if (kind == "frames") {
    py::array frames = contiguous(result["frames"]);
    out.path = exportMp4(viewOf(frames), model_->fps(request), dir);
  } else if (kind == "av") {
    py::array frames = contiguous(result["frames"]);
    py::array audio = contiguous(result["audio"]);
    AudioVideo av;
    av.frames = viewOf(frames);
    av.audio = viewOf(audio);
    av.samplingRate = result["sampling_rate"].cast<int>();
    av.fps = result["fps"].cast<double>();
    av.pixelFormat = result["pixel_format"].cast<std::string>();
    out.path = exportMp4WithAudio(av, dir);
  } else if (kind == "images") {
    for (const auto& image : result["images"]) {
      out.images.push_back(image.cast<std::string>());
    }
  } else {
    throw std::runtime_error("adapter returned an unknown result kind '" +
                             kind + "'");
  }
  TT_LOG_INFO("[DiT] {} task {} exported in {:.1f} s", model_->spec().logName,
              request.taskId, secondsSince(started));
  return out;
}

void DiTRunner::infer(const DiTRequest& request) {
  py::dict result = callAdapter(request);
  // Only rank 0 keeps a video; a file an adapter wrote itself is dropped.
  if (py::str(result["kind"]).cast<std::string>() == "path") {
    deleteIfFile(result["path"].cast<std::string>());
  }
}

py::dict DiTRunner::callAdapter(const DiTRequest& request) {
  if (!python_->adapter) {
    throw std::runtime_error("warmup() must finish before a request is run");
  }
  validate(request);
  const auto started = std::chrono::steady_clock::now();
  py::dict result =
      python_->adapter.attr("call")(toPython(model_->buildCall(request)));
  TT_LOG_INFO("[DiT] {} task {} inference done in {:.1f} s",
              model_->spec().logName, request.taskId, secondsSince(started));
  return result;
}

void DiTRunner::close() {
  python_->adapter = py::object();
  if (!python_->mesh) return;
  py::object mesh = std::move(python_->mesh);
  python_->mesh = py::object();
  py::module_::import("ttnn").attr("close_mesh_device")(mesh);
  TT_LOG_INFO("[DiT] Closed mesh device");
}

bool DiTRunner::meshIsDistributed() const {
  if (!python_->mesh) return false;
  // Strict `is True`: only a real distributed environment wraps the runner.
  py::object distributed =
      py::module_::import("ttnn").attr("using_distributed_env")();
  return py::isinstance<py::bool_>(distributed) && distributed.cast<bool>();
}

std::unique_ptr<IDiTRunner> createOpenRunner(const RunnerConfig& config) {
  auto runner = std::make_unique<DiTRunner>(createModel(config));
  runner->openDevice();
  try {
    // Opening the mesh creates the distributed context; only now is it known
    // whether the mesh spans several hosts.
    if (runner->meshIsDistributed()) {
      TT_LOG_INFO(
          "[DiT] Multihost mesh: rank 0 runs in lockstep with the others");
      return std::make_unique<LockstepRunner>(std::move(runner),
                                              std::make_shared<MpiComm>());
    }
  } catch (...) {
    // The caller never gets a runner to stop(), so the open mesh would keep the
    // chips claimed: close it here.
    try {
      if (runner) runner->close();  // null once LockstepRunner took it
    } catch (...) {
    }
    throw;
  }
  return runner;
}

int runFollowerRank(int rank) {
  try {
    auto runner =
        std::make_unique<DiTRunner>(createModel(RunnerConfig::fromEnv()));
    return Follower(rank, std::move(runner), std::make_shared<MpiComm>()).run();
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[DiTMultihost] Rank {}: cannot start: {}", rank, e.what());
    return 1;
  }
}

}  // namespace tt::dit
