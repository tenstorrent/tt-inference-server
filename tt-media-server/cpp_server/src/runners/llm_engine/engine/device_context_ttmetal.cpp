#include "llm_engine/engine/device_context_ttmetal.hpp"
#include "llm_engine/engine/debug.hpp"

#include <cstdlib>
#include <exception>
#include <stdexcept>

#if defined(TT_METAL_RUNTIME_ROOT_DEFAULT)
static void ensure_tt_metal_runtime_root() {
  if (std::getenv("TT_METAL_RUNTIME_ROOT") == nullptr) {
    setenv("TT_METAL_RUNTIME_ROOT", TT_METAL_RUNTIME_ROOT_DEFAULT, 1);
  }
}
#else
static void ensure_tt_metal_runtime_root() {}
#endif

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include <memory>

namespace llm_engine {

namespace {

/** Page size must be PCIe-aligned (e.g. 64). One page = one DecodeEntry (first 12 bytes) + padding. Must match model_runner and loopback kernel. */
constexpr uint32_t kPageSize = 64;
constexpr uint32_t kDataSizeOnePage = 64;
constexpr uint32_t kFifoSize = 2048;
/** Iterations for streaming: kernel runs this many times (one page per iteration). Enqueued once at startup. */
constexpr uint32_t kNumIterationsStreaming = 65536;
const char* kLoopbackKernelPath =
    "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp";

struct TTMetalDecodeContext {
  using MeshDevice = tt::tt_metal::distributed::MeshDevice;
  using H2DSocket = tt::tt_metal::distributed::H2DSocket;
  using D2HSocket = tt::tt_metal::distributed::D2HSocket;
  using MeshCoreCoord = tt::tt_metal::distributed::MeshCoreCoord;
  using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
  using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;
  using MeshShape = tt::tt_metal::distributed::MeshShape;
  using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;
  using MeshWorkload = tt::tt_metal::distributed::MeshWorkload;

  std::shared_ptr<MeshDevice> mesh_device;
  std::unique_ptr<H2DSocket> h2d_socket;
  std::unique_ptr<D2HSocket> d2h_socket;
  MeshCoreCoord socket_core;
};

}  // namespace

void* create_ttmetal_decode_context_and_config(Config* config) {
  ensure_tt_metal_runtime_root();

  try {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    if (GetNumAvailableDevices() == 0) {
      LLM_ENGINE_LOG("device_context_ttmetal") << "No devices; skipping real sockets."
                                               << std::endl;
      return nullptr;
    }

    auto ctx = new TTMetalDecodeContext{};
    ctx->mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    ctx->socket_core = MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(0, 0));

    ctx->h2d_socket = std::make_unique<H2DSocket>(
        ctx->mesh_device, ctx->socket_core, BufferType::L1, kFifoSize, H2DMode::HOST_PUSH);
    ctx->h2d_socket->set_page_size(kPageSize);

    ctx->d2h_socket = std::make_unique<D2HSocket>(ctx->mesh_device, ctx->socket_core, kFifoSize);
    ctx->d2h_socket->set_page_size(kPageSize);

    auto loopback_program = CreateProgram();
    CreateKernel(
        loopback_program,
        kLoopbackKernelPath,
        ctx->socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(ctx->h2d_socket->get_config_buffer_address()),
                static_cast<uint32_t>(ctx->d2h_socket->get_config_buffer_address()),
                kPageSize,
                kDataSizeOnePage,
                kNumIterationsStreaming,
                static_cast<uint32_t>(false),
            }});
    MeshWorkload mesh_workload;
    mesh_workload.add_program(
        MeshCoordinateRange(ctx->socket_core.device_coord), std::move(loopback_program));
    EnqueueMeshWorkload(ctx->mesh_device->mesh_command_queue(), mesh_workload, false);

    config->mesh_device = ctx->mesh_device.get();
    config->h2d_socket = ctx->h2d_socket.get();
    config->d2h_socket = ctx->d2h_socket.get();
    config->enqueue_one_page = nullptr;
    config->enqueue_one_page_ctx = nullptr;

    LLM_ENGINE_LOG("device_context_ttmetal")
        << "Opened 1x1 mesh, H2D/D2H sockets (page_size=" << kPageSize << "), loopback kernel enqueued once ("
        << kNumIterationsStreaming << " iterations)." << std::endl;
    return ctx;
  } catch (const std::exception& e) {
    LLM_ENGINE_LOG("device_context_ttmetal") << "tt-metal device open failed: " << e.what()
                                             << "." << std::endl;
    return nullptr;
  }
}

void destroy_ttmetal_decode_context(void* ctx) {
  if (!ctx) return;
  auto* c = static_cast<TTMetalDecodeContext*>(ctx);
  Finish(c->mesh_device->mesh_command_queue());
  c->mesh_device->close();
  delete c;
  LLM_ENGINE_LOG("device_context_ttmetal") << "Device closed." << std::endl;
}

}  // namespace llm_engine
