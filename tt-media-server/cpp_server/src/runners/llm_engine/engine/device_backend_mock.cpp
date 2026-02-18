#include "llm_engine/engine/device_backend.hpp"
#include "llm_engine/engine/sequence.hpp"

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>

namespace llm_engine {

namespace {

struct DecodeResultPage {
  std::vector<char> data;
};

class MockDeviceBackend : public IDeviceBackend {
 public:
  explicit MockDeviceBackend(const Config&) {}

  void init() override {}

  void write(const void* data, uint32_t num_pages) override {
    if (num_pages == 0) return;
    const char* p = static_cast<const char*>(data);
    size_t page_sz = Sequence::page_size();
    DecodeResultPage page;
    page.data.resize(page_sz);
    std::memcpy(page.data.data(), p, page_sz);
    int64_t* token_ptr = reinterpret_cast<int64_t*>(page.data.data() + SequenceID::kSerializedSize);
    *token_ptr += 1;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(page));
    }
    cv_.notify_one();
  }

  bool read(void* data, uint32_t num_pages) override {
    if (num_pages == 0) return false;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
    if (stop_ && queue_.empty()) return false;
    if (!queue_.empty()) {
      DecodeResultPage page = std::move(queue_.front());
      queue_.pop();
      lock.unlock();
      std::memcpy(data, page.data.data(), page.data.size());
      return true;
    }
    return false;
  }

  void terminate() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<DecodeResultPage> queue_;
  bool stop_ = false;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_mock(const Config& config) {
  return std::make_unique<MockDeviceBackend>(config);
}

}  // namespace llm_engine
