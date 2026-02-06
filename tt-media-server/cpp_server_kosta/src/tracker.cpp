#include "tracker.hpp"

Tracker &Tracker::getInstance() {
  static Tracker instance;
  return instance;
}

int Tracker::put(drogon::ResponseStreamPtr stream) {
  int id = next_id.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->pending_requests.insert_or_assign(id, PendingRequest{std::move(stream)});
  }
  total_requests.fetch_add(1);
  return id;
}

PendingRequest* Tracker::get(int key) {
  std::lock_guard<std::mutex> lock(this->mutex);
  auto it = this->pending_requests.find(key);
  if (it == this->pending_requests.end()) {
    return nullptr;
  }
  return &it->second;
}

void Tracker::remove(int key) {
  std::lock_guard<std::mutex> lock(this->mutex);
  if (this->pending_requests.erase(key) > 0) {
    completed_requests.fetch_add(1);
  }
}

size_t Tracker::getPendingCount() const {
  std::lock_guard<std::mutex> lock(this->mutex);
  return pending_requests.size();
}
