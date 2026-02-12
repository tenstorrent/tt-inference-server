#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/device_context_ttmetal.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llm_engine {

namespace {

struct DecodeEntry {
  uint32_t token_id;
  uint32_t user_id;
  uint32_t position_id;
};
constexpr uint32_t kWordsPerEntry = 3;
constexpr uint32_t kBytesPerEntry = kWordsPerEntry * sizeof(uint32_t);
static_assert(sizeof(DecodeEntry) == kBytesPerEntry, "DecodeEntry must be 12 bytes");

/** Page size must be PCIe-aligned (e.g. 64). One page = one DecodeEntry (first 12 bytes) + padding. Must match device_context_ttmetal and loopback kernel. */
constexpr uint32_t kPageSize = 64;
constexpr uint32_t kPageSizeWords = kPageSize / sizeof(uint32_t);
static_assert(kPageSize >= kBytesPerEntry, "page must fit one DecodeEntry");

}  // namespace

class ModelRunnerSocketSim : public IModelRunner {
 public:
  explicit ModelRunnerSocketSim(const Config& config);
  ~ModelRunnerSocketSim() override { exit(); }

  void set_on_tokens_callback(OnTokensCallback on_tokens) override;
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override;
  void exit() override;

 private:
  void receiver_loop();

  Config config_;
  int eos_;
  void* device_ctx_ = nullptr;
  OnTokensCallback on_tokens_callback_;

  std::thread receiver_thread_;
  std::atomic<bool> receiver_shutdown_{false};
};

ModelRunnerSocketSim::ModelRunnerSocketSim(const Config& config)
    : config_(config), eos_(config.eos) {
  device_ctx_ = create_ttmetal_decode_context_and_config(&config_);
  if (!device_ctx_ || !config_.h2d_socket || !config_.d2h_socket) {
    throw std::runtime_error("model_runner: tt-metal device and H2D/D2H sockets required");
  }
  receiver_thread_ = std::thread{&ModelRunnerSocketSim::receiver_loop, this};
  LLM_ENGINE_LOG("model_runner") << "H2D/D2H sockets ready (long-running receiver thread)"
                                << std::endl;
}

void ModelRunnerSocketSim::receiver_loop() {
  auto* d2h = static_cast<tt::tt_metal::distributed::D2HSocket*>(config_.d2h_socket);
  while (!receiver_shutdown_.load()) {
    uint32_t recv_buf[kPageSizeWords] = {};
    d2h->read(recv_buf, 1);
    d2h->barrier();

    int64_t token_id = static_cast<int64_t>(recv_buf[0]) + 1;
    int user_id = static_cast<int>(recv_buf[1]);
    LLM_ENGINE_LOG("model_runner") << "decode user_id=" << user_id
                                  << " -> token=" << token_id << std::endl;
    if (on_tokens_callback_) on_tokens_callback_({{token_id, user_id}});
  }
}

void ModelRunnerSocketSim::set_on_tokens_callback(OnTokensCallback on_tokens) {
  on_tokens_callback_ = std::move(on_tokens);
}

void ModelRunnerSocketSim::run(const std::vector<Sequence*>& seqs, bool is_prefill) {
  if (is_prefill) {
    LLM_ENGINE_LOG("model_runner") << "prefill batch_size=" << seqs.size()
                                   << std::endl;
    std::vector<TokenEntry> entries;
    entries.reserve(seqs.size());
    int64_t token = seqs[0]->last_token + 1;
    for (Sequence* s : seqs) {
      entries.emplace_back(token, s->seq_id);
    }
    if (on_tokens_callback_) on_tokens_callback_(std::move(entries));
    return;
  }

  if (seqs.empty()) return;

  Sequence* s = seqs[0];
  uint32_t h2d_buf[kPageSizeWords] = {};
  h2d_buf[0] = static_cast<uint32_t>(s->last_token & 0xFFFFFFFFu);
  h2d_buf[1] = static_cast<uint32_t>(s->seq_id & 0xFFFFFFFFu);
  h2d_buf[2] = static_cast<uint32_t>(s->size() & 0xFFFFFFFFu);
  auto* h2d = static_cast<tt::tt_metal::distributed::H2DSocket*>(config_.h2d_socket);
  h2d->write(h2d_buf, 1);
}

void ModelRunnerSocketSim::exit() {
  if (receiver_thread_.joinable()) {
    receiver_shutdown_.store(true);
    receiver_thread_.join();
  }
  if (device_ctx_) {
    LLM_ENGINE_LOG("model_runner") << "exit (destroy device context)" << std::endl;
    destroy_ttmetal_decode_context(device_ctx_);
    device_ctx_ = nullptr;
  }
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config) {
  return std::make_unique<ModelRunnerSocketSim>(config);
}

ModelRunnerStub::ModelRunnerStub(const Config& config)
    : config_(config), eos_(config.eos) {}

void ModelRunnerStub::set_on_tokens_callback(OnTokensCallback on_tokens) {
  on_tokens_callback_ = std::move(on_tokens);
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs, bool is_prefill) {
  if (seqs.empty()) return;
  std::vector<TokenEntry> entries;
  entries.reserve(seqs.size());
  int64_t token = seqs[0]->last_token + 1;
  for (Sequence* s : seqs) {
    entries.emplace_back(token, s->seq_id);
  }
  if (on_tokens_callback_) on_tokens_callback_(std::move(entries));
}

void ModelRunnerStub::exit() {}

}  // namespace llm_engine