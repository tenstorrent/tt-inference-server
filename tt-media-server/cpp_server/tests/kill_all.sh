#!/bin/bash
# Nuclear option - kill all tt_consumer and tt_media_server processes

echo "Finding all processes..."
ps aux | grep -E "[t]t_consumer|[t]t_media_server"

echo ""
echo "Killing all processes..."

# Get all PIDs
PIDS=$(ps aux | grep -E "[t]t_consumer|[t]t_media_server" | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "No processes found to kill"
else
  echo "Killing PIDs: $PIDS"
  echo "$PIDS" | xargs kill -9 2>/dev/null || true
  sleep 1
  echo ""
  echo "Verification - remaining processes:"
  ps aux | grep -E "[t]t_consumer|[t]t_media_server" || echo "✓ All processes killed"
fi
