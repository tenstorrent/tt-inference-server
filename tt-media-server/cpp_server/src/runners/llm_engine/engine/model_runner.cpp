#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

namespace llm_engine {

constexpr uint32_t kPageSize = 128;
constexpr uint32_t kFifoSize = 2048;
constexpr uint32_t kNumIterations = 64;

void DecodeQueue::push(const DecodeResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.push_back(result);
}

std::vector<DecodeResult> DecodeQueue::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<DecodeResult> out;
  out.swap(pending_);
  return out;
}

ModelRunnerStub::ModelRunnerStub(const Config& config, DecodeCallback callback)
    : config_(config),
      dummy_token_((config.eos == 0) ? 1 : 0),
      decode_callback_(std::move(callback)),
      reader_thread_([this] { reader_loop(); }) {
        std::cout << "[host_io] create_unit_mesh..." << std::endl;
        
        try {
        mesh_device_ = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        } catch (const std::exception& e) {
            std::cerr << "[host_io] create_unit_mesh failed: " << e.what() << std::endl;
            throw;
        }
        std::cout << "[host_io] create_unit_mesh done" << std::endl;
    
        tt::tt_metal::distributed::MeshCoordinate device_coord{0, 0};
        tt::tt_metal::CoreCoord core_coord{0, 0};
        tt::tt_metal::distributed::MeshCoreCoord socket_core{device_coord, core_coord};

        std::cout << "[host_io] creating H2DSocket..." << std::endl;
        h2d_socket_ = std::make_unique<tt::tt_metal::distributed::H2DSocket>(
            mesh_device_,
            socket_core,
            tt::tt_metal::BufferType::L1,
            kFifoSize,
            tt::tt_metal::distributed::H2DMode::HOST_PUSH);
        h2d_socket_->set_page_size(kPageSize);
        std::cout << "[host_io] H2DSocket done" << std::endl;
    
        std::cout << "[host_io] creating D2HSocket..." << std::endl;
        d2h_socket_ = std::make_unique<tt::tt_metal::distributed::D2HSocket>(
            mesh_device_, socket_core, kFifoSize);
        d2h_socket_->set_page_size(kPageSize);
        std::cout << "[host_io] D2HSocket done" << std::endl;
      }

ModelRunnerStub::~ModelRunnerStub() {
  exit();
}

void ModelRunnerStub::reader_loop() {
  std::vector<char> output(Sequence::h2d_size());
  while (!stop_.load(std::memory_order_relaxed)) {
    d2h_socket_->read(output.data(), output.size());
    Sequence* seq = Sequence::from_h2d_input(output);
    decode_callback_(DecodeResult{seq->seq_id, seq->last_token});
  }
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    for (Sequence* seq : seqs) {
      decode_callback_({seq->seq_id, seq->last_token + 1});
    }
  } else {
    {
      std::lock_guard<std::mutex> lock(batch_mutex_);
      batch_queue_.push_back(seqs);
    }
    h2d_socket_->write(seqs[0]->to_h2d_input().data(), seqs[0]->to_h2d_input().size());
  }
}

void ModelRunnerStub::exit() {
  if (stop_.exchange(true)) return;
  if (reader_thread_.joinable()) reader_thread_.join();
  mesh_device_->mesh_command_queue().finish();
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  return std::make_unique<ModelRunnerStub>(config, std::move(callback));
}

}  // namespace llm_engine
