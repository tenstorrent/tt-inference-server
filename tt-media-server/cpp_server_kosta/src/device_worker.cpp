#include "device_worker.hpp"
#include <glog/logging.h>

void DeviceWorker::start() {
  LOG(INFO) << "Worker " << workerId << ": started (task_queue" << queueIndex << ")";
  for (size_t i = 0; i < maxBatchSize; i++) {
      boost::asio::post(pool, [this, i]() { // pass req to thread pool
        while (true) {
          domain::Request req;
          unsigned int priority;
          message_queue::size_type recvd_size;

          this->taskQueue.receive(&req, sizeof(req), recvd_size, priority); // -> blocking
          
          // Phase 2: Use batched streaming for reduced IPC overhead
          this->runner.run_streaming_batched(req, [this](const domain::BatchedResponse &batch) {
            this->responseQueue.send(&batch, sizeof(batch), 0);
          });
        }
    });
  }
  pool.join();
}
