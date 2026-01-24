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
TCP_PORT=8081

# Common pipeline: scale and convert to BGRx 640x640 for YOLOv8s
YOLO_PROCESS="queue ! videoconvert ! videoscale ! video/x-raw,format=BGRx,width=640,height=640 ! yolov8s"

# Output: MJPEG stream (TCP for VLC, HTTP wrapper for browser)
STREAM_OUTPUT="videoconvert ! jpegenc quality=85 ! multipartmux boundary=frame ! tcpserversink host=0.0.0.0 port=$TCP_PORT"

# Function to start HTTP wrapper for browser access
start_http_wrapper() {
    echo "Starting HTTP server on port $STREAM_PORT..."
    python3 /app/http_stream.py &
    sleep 1
}

echo "============================================"
echo "  GStreamer YOLOv8s Demo on Tenstorrent"
echo "============================================"

# Check plugin
if ! gst-inspect-1.0 yolov8s > /dev/null 2>&1; then
    echo "ERROR: yolov8s plugin not found!"
    exit 1
fi
echo "Plugin: yolov8s loaded"

# Show help
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo ""
    echo "Usage: docker run ... tt-gstreamer-yolov8 [MODE]"
    echo ""
    echo "  MODES:"
    echo "    (no args)       - Demo with built-in traffic video"
    echo "    webcam-server   - Browser webcam mode (click Webcam tab)"
    echo "    stream-file     - Custom video file"
    echo "    stream-rtsp     - RTSP IP camera"
    echo "    stream-webcam   - USB webcam on server"
    echo "    parallel-8device  - 8-device parallel stream (8 video grid)"
    echo "    benchmark-8device - 8-device throughput test (~980 FPS)"
    echo ""
    echo "  Examples:"
    echo "    docker run --network host ... tt-gstreamer-yolov8"
    echo "    docker run --network host ... tt-gstreamer-yolov8 webcam-server"
    echo "    docker run --network host ... tt-gstreamer-yolov8 stream-rtsp rtsp://camera/stream"
    echo ""
    exit 0
fi

# Default: run demo with built-in video (no arguments needed!)
if [ $# -eq 0 ]; then
    echo ""
    echo "Starting YOLOv8s Demo with built-in video..."
    echo "  Browser: http://localhost:$STREAM_PORT"
    echo "  (Model compiles on first run ~2 min)"
    echo ""
    start_http_wrapper
    # Play video (will stop when video ends)
    gst-launch-1.0 -v filesrc location="/app/demo/city_traffic.mp4" ! \
        decodebin ! videoconvert ! videoscale ! videorate ! \
        "video/x-raw,width=640,height=640,framerate=30/1" ! \
        $YOLO_PROCESS ! $STREAM_OUTPUT
    exit 0
fi

MODE=$1
shift

case $MODE in
    "stream-test")
        echo ""
        echo "Streaming test pattern..."
        echo "  Browser: http://localhost:$STREAM_PORT"
        echo ""
        start_http_wrapper
        gst-launch-1.0 -v videotestsrc ! \
            "video/x-raw,width=640,height=480,framerate=30/1,format=UYVY" ! \
            $YOLO_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-file")
        VIDEO_PATH=$1
        echo ""
        echo "Streaming video: $VIDEO_PATH"
        echo "  Browser: http://localhost:$STREAM_PORT"
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
        echo "Streaming from RTSP: $RTSP_URL"
        echo "  Browser: http://localhost:$STREAM_PORT"
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
        echo "Streaming from USB webcam: $DEVICE"
        echo "  Browser: http://localhost:$STREAM_PORT"
        echo ""
        start_http_wrapper
        gst-launch-1.0 -v v4l2src device=$DEVICE ! \
            "video/x-raw,width=640,height=480,framerate=30/1" ! \
            $YOLO_PROCESS ! $STREAM_OUTPUT
        ;;

    "benchmark-8device")
        echo "Running 8-device parallel benchmark..."
        python3 /app/test_8device_parallel.py "$@"
        ;;

    "parallel-8device")
        # Collect all remaining args as video paths
        VIDEO_ARGS=""
        if [ $# -gt 0 ]; then
            VIDEO_ARGS="$@"
        else
            VIDEO_ARGS="/app/demo/city_traffic.mp4"
        fi
        echo ""
        echo "Starting 8-Device Parallel Streaming Demo..."
        echo "  Videos: $VIDEO_ARGS"
        echo "  Browser: http://localhost:$STREAM_PORT"
        echo "  (Shows 8 video outputs in grid - one per device)"
        echo ""
        python3 /app/parallel_8device_stream.py $VIDEO_ARGS --port $STREAM_PORT
        ;;

    "webcam-server")
        echo ""
        echo "Starting Browser Webcam Server..."
        echo "  Browser: http://localhost:$STREAM_PORT"
        echo "  Click 'Webcam' tab"
        echo "  (Model compiles on first run ~2 min)"
        echo ""
        start_http_wrapper
        python3 /app/websocket_server.py "$@"
        ;;

    "custom")
        echo "Running custom command..."
        exec "$@"
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Run with --help for usage"
        exit 1
        ;;
esac
