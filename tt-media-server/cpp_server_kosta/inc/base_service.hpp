#pragma once
#include "api.hpp"
#include "scheduler.hpp"
#include "tracker.hpp"
#include <drogon/drogon.h>

class BaseService {
private:
  Scheduler &scheduler;
  Tracker &tracker;
  BaseService()
      : scheduler(Scheduler::getInstace()), tracker(Tracker::getInstance()) {}

public:
  static BaseService &getInstace();
  void process(const api::Request &request, drogon::ResponseStreamPtr stream);
};
