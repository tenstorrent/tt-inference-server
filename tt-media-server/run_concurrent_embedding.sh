#!/bin/bash

# Load testing script for embedding endpoint
# Usage: ./load_test.sh [requests_per_worker]

# Configuration
REQUESTS=${1:-10}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="embedding_load_test_${TIMESTAMP}.log"

# Create input text with exactly 255 "hello"s
HELLO_128="$(printf 'hello %.0s' {1..126})"
HELLO_256="$(printf 'hello %.0s' {1..250})"
HELLO_512="$(printf 'hello %.0s' {1..510})"
HELLO_1024="$(printf 'hello %.0s' {1..1020})"
HELLO_2048="$(printf 'hello %.0s' {1..2040})"
HELLO_4096="$(printf 'hello %.0s' {1..4090})"

# Start time
START_TIME=$(date +%s.%N)

# Initialize output file with header
echo "Embedding Load Test Results" > "$OUTPUT_FILE"
echo "Started at: $(date)" >> "$OUTPUT_FILE"
echo "Requests per worker: $REQUESTS" >> "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Array to store background process PIDs
declare -a pids

for j in $(seq 1 $REQUESTS); do
    (
        response=$(curl -s -X POST "http://localhost:8012/v1/embeddings" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer your-secret-key" \
        -d "{\"input\": \"${HELLO_128}\"}" \
        -w "HTTPSTATUS:%{http_code};TIME:%{time_total};SIZE:%{size_download}" )
        
        # Parse curl output
        http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        time_total=$(echo "$response" | grep -o "TIME:[0-9.]*" | cut -d: -f2)
        
        # Check if successful
        if [ "$http_code" = "200" ]; then
        status="✓"
        else
        status="x"
        fi
        
        result="Request $j: $status HTTP $http_code - ${time_total}s"
        echo "$result"
        echo "$result" >> "$OUTPUT_FILE"
    ) &
    pids+=($!)  # Store the background process PID
done

# Wait for all background jobs to complete
echo "Waiting for $REQUESTS concurrent requests to complete..."
wait

echo "All requests completed!"
echo "All requests completed!" >> "$OUTPUT_FILE"

# End time
END_TIME=$(date +%s.%N)
TOTAL_TIME=$(awk "BEGIN {print $END_TIME - $START_TIME}")

echo "----------------------------------------"
echo "Load test completed!"
echo "Total time: ${TOTAL_TIME} seconds"

# Write summary to output file
echo "" >> "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"
echo "Load test completed!" >> "$OUTPUT_FILE"
echo "Total time: ${TOTAL_TIME} seconds" >> "$OUTPUT_FILE"
echo "Completed at: $(date)" >> "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
