#!/bin/bash
# Diagnose where Kafka latency is coming from

echo "=== Kafka Latency Diagnostics ==="
echo ""

# 1. Check Kafka is local
echo "1. Checking Kafka broker location..."
BROKER=$(netstat -tulpn 2>/dev/null | grep 9092 || ss -tulpn 2>/dev/null | grep 9092)
if [[ $BROKER == *"127.0.0.1"* ]] || [[ $BROKER == *"0.0.0.0"* ]]; then
  echo "   ✓ Kafka is on localhost (good)"
else
  echo "   ✗ Kafka might be remote - this adds network latency"
fi
echo ""

# 2. Check topic configuration
echo "2. Checking topic configuration..."
kafka-topics --bootstrap-server localhost:9092 --describe --topic session-offload 2>/dev/null
echo ""

# 3. Check consumer group lag
echo "3. Checking consumer group status..."
kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group migration-workers 2>/dev/null || echo "   (Group not active - start consumers first)"
echo ""

# 4. Benchmark raw Kafka performance
echo "4. Benchmarking raw Kafka producer performance..."
echo "   Sending 100 small messages with acks=1..."
kafka-producer-perf-test --topic session-offload \
  --num-records 100 \
  --record-size 100 \
  --throughput -1 \
  --producer-props bootstrap.servers=localhost:9092 \
    acks=1 linger.ms=0 batch.size=1 2>/dev/null | grep -E "records sent|Throughput|latency"
echo ""

echo "5. Recommendations:"
echo "   - If broker avg latency > 10ms: Check server.properties disk flush settings"
echo "   - If consumer lag > 0: Consumers can't keep up, add more consumers"
echo "   - If topic has only 1 partition: Create topic with 3+ partitions for parallelism"
echo ""
echo "6. Kafka broker tuning (add to server.properties):"
echo "   log.flush.interval.messages=1"
echo "   log.flush.interval.ms=1"
echo "   num.network.threads=8"
echo "   socket.send.buffer.bytes=131072"
echo ""
echo "7. tt_media_server_cpp / tt_consumer client env (kafka_client.cpp):"
echo "   KAFKA_COMPRESSION_TYPE=none"
echo "   KAFKA_PRODUCER_QUEUE_BUFFERING_MAX_MESSAGES=10000   (default)"
echo "   KAFKA_PRODUCER_QUEUE_BUFFERING_MAX_KBYTES=4096      (default)"
echo "   KAFKA_CONSUMER_QUEUED_MIN_MESSAGES=1                (default)"
echo "   KAFKA_CONSUMER_QUEUED_MAX_MESSAGES_KBYTES=1024      (default)"
echo "   SESSION_OFFLOAD_KAFKA_MIN_INTERVAL_MS=50-200      (producer burst / tail latency)"
echo ""
