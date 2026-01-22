#!/bin/bash
# GStreamer YOLOv8s Demo Entrypoint
set -e

# Activate the GStreamer-compatible venv
source /home/container_app_user/python_env_gst/bin/activate
cd /home/container_app_user/tt-metal

# Set environment
export GST_PLUGIN_PATH=/app/gstreamer_plugins/plugins
export PYTHONPATH=/home/container_app_user/tt-metal:$PYTHONPATH
export GST_DEBUG=${GST_DEBUG:-1}

# Clear GStreamer cache
rm -f ~/.cache/gstreamer-1.0/registry.x86_64.bin 2>/dev/null || true

# Create __init__.py if missing
touch /home/container_app_user/tt-metal/tests/__init__.py 2>/dev/null || true
touch /home/container_app_user/tt-metal/tests/ttnn/__init__.py 2>/dev/null || true

STREAM_PORT=${STREAM_PORT:-8080}

echo "============================================"
echo "  GStreamer YOLOv8s Demo on Tenstorrent"
echo "============================================"

# Check plugin
if ! gst-inspect-1.0 yolov8s > /dev/null 2>&1; then
    echo "ERROR: yolov8s plugin not found!"
    exit 1
fi
echo "✅ yolov8s plugin loaded"

# Default: show usage
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage:"
    echo ""
    echo "  LIVE STREAMING (view in browser at http://host:$STREAM_PORT):"
    echo "    stream-test     - Test pattern"
    echo "    stream-file     - Video file"
    echo "    stream-rtsp     - RTSP camera"
    echo "    stream-webcam   - USB webcam"
    echo ""
    echo "  Examples:"
    echo "    docker run -p 8080:8080 ... tt-gstreamer-yolov8 stream-test"
    echo "    docker run -p 8080:8080 -v /videos:/videos ... tt-gstreamer-yolov8 stream-file /videos/input.mp4"
    echo "    docker run -p 8080:8080 ... tt-gstreamer-yolov8 stream-rtsp rtsp://user:pass@camera:554/stream"
    echo ""
    exit 0
fi

MODE=$1
shift

# Common pipeline: scale and convert to BGRx 640x640 for YOLOv8s
YOLO_PROCESS="queue ! videoconvert ! videoscale ! video/x-raw,format=BGRx,width=640,height=640 ! yolov8s"

# Output: MJPEG stream (TCP for VLC, HTTP wrapper for browser)
TCP_PORT=8081
STREAM_OUTPUT="videoconvert ! jpegenc quality=85 ! multipartmux boundary=frame ! \
    tcpserversink host=0.0.0.0 port=$TCP_PORT"

# Function to start HTTP wrapper for browser access
start_http_wrapper() {
    echo "Starting HTTP server on port $STREAM_PORT..."
    python3 /app/http_stream.py &
    sleep 1
}

case $MODE in
    "stream-test")
        echo ""
        echo "🎥 Streaming test pattern"
        echo "   Browser: http://HOST:$STREAM_PORT"
        echo "   VLC: vlc tcp://HOST:$TCP_PORT"
        echo "   (Model compiles on first run ~2 min)"
        echo ""
        start_http_wrapper
        gst-launch-1.0 -v videotestsrc ! \
            "video/x-raw,width=640,height=480,framerate=30/1,format=UYVY" ! \
            $YOLO_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-file")
        VIDEO_PATH=$1
        echo ""
        echo "🎥 Streaming video: $VIDEO_PATH"
        echo "   Browser: http://HOST:$STREAM_PORT"
        echo "   VLC: vlc tcp://HOST:$TCP_PORT"
        echo ""
        start_http_wrapper
        gst-launch-1.0 -v filesrc location="$VIDEO_PATH" ! \
            decodebin ! videoconvert ! videoscale ! videorate ! \
            "video/x-raw,width=640,height=640,framerate=30/1" ! \
            $YOLO_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-rtsp")
        RTSP_URL=$1
        echo ""
        echo "🎥 Streaming from RTSP: $RTSP_URL"
        echo "   Browser: http://HOST:$STREAM_PORT"
        echo ""
        start_http_wrapper
        gst-launch-1.0 -v rtspsrc location="$RTSP_URL" latency=0 protocols=tcp ! \
            rtph264depay ! h264parse ! avdec_h264 ! \
            videoconvert ! videoscale ! videorate ! \
            "video/x-raw,width=640,height=640,framerate=30/1" ! \
            $YOLO_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-webcam")
        DEVICE=${1:-/dev/video0}
        echo ""
        echo "🎥 Streaming from webcam: $DEVICE"
        echo "   Browser: http://HOST:$STREAM_PORT"
        echo ""
        start_http_wrapper
        gst-launch-1.0 -v v4l2src device=$DEVICE ! \
            "video/x-raw,width=640,height=480,framerate=30/1" ! \
            $YOLO_PROCESS ! $STREAM_OUTPUT
        ;;

    "benchmark")
        echo "Running benchmark (no output)..."
        gst-launch-1.0 -v videotestsrc num-buffers=100 ! \
            "video/x-raw,width=640,height=480,framerate=30/1,format=UYVY" ! \
            $YOLO_PROCESS ! fakesink sync=false
        ;;

    "custom")
        echo "Running custom command..."
        exec "$@"
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
