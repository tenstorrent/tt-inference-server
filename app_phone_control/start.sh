#!/usr/bin/env bash
# TT-Home Phone Control PWA — Launcher
# Usage: bash start.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8081}"

echo "============================================"
echo "  TT-Home Phone Control PWA"
echo "============================================"
echo ""

# Check Python deps
for pkg in fastapi uvicorn httpx; do
  python3 -c "import $pkg" 2>/dev/null || {
    echo "[!] Missing $pkg — installing..."
    pip3 install --quiet "$pkg"
  }
done

echo "[1/2] Checking TT-Home on port 8080..."
if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
  echo "       TT-Home is reachable ✅"
else
  echo "       TT-Home not detected (Personal mode will show offline)"
fi

echo "[2/2] Starting PWA server on port $PORT..."
echo ""
echo "  Open on your phone (same WiFi):"

# Print all non-loopback IPs
for ip in $(hostname -I 2>/dev/null); do
  echo "    → http://$ip:$PORT"
done
echo ""
echo "  Tap the share button → 'Add to Home Screen' for app-like experience"
echo "============================================"
echo ""

cd "$SCRIPT_DIR"
exec python3 -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --log-level info
