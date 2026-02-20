#include "runners/llm_runner/device_backend.hpp"
#include "runners/llm_runner/sequence.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>

namespace llm_engine {

namespace {

class MockDeviceBackend : public IDeviceBackend {
 public:
  explicit MockDeviceBackend(const Config&) {}

  void init() override {}

  void write(const Sequence& seq) override {
    std::lock_guard<std::mutex> lock(work_mutex_);
    work_queue_.push(DecodeResult{seq.task_id, seq.last_token + 1});
    cv_.notify_one();
  }

  bool read(DecodeResult* result) override {
    std::unique_lock<std::mutex> lock(work_mutex_);
    cv_.wait(lock, [this] { return stop_ || !work_queue_.empty(); });
    if (stop_ && work_queue_.empty()) return false;
    if (!work_queue_.empty()) {
      *result = std::move(work_queue_.front());
      work_queue_.pop();
      return true;
    }
    return false;
  }

  void terminate() override {
    std::lock_guard<std::mutex> lock(work_mutex_);
    stop_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex work_mutex_;
  std::queue<DecodeResult> work_queue_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_mock(const Config& config) {
  return std::make_unique<MockDeviceBackend>(config);
}

}  // namespace llm_engine
