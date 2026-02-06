#pragma once
#define GLOG_USE_GLOG_EXPORT
#include <atomic>
#include <drogon/drogon.h>
#include <unordered_map>
#include <mutex>
#include <string>

struct PendingRequest {
  drogon::ResponseStreamPtr stream;
  std::string accumulated_output;  // Buffer for accumulating SSE data
};

class Tracker {
private:
  mutable std::mutex mutex;
  std::unordered_map<int, PendingRequest> pending_requests;
  std::atomic<int> next_id{0};
  std::atomic<uint64_t> total_requests{0};
  std::atomic<uint64_t> completed_requests{0};

public:
  static Tracker &getInstance();
  int put(drogon::ResponseStreamPtr stream);
  PendingRequest* get(int key);
  void remove(int key);
  
  // Metrics
  size_t getPendingCount() const;
  uint64_t getTotalRequests() const { return total_requests.load(); }
  uint64_t getCompletedRequests() const { return completed_requests.load(); }
};
