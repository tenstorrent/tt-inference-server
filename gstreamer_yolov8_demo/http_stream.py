#!/usr/bin/env python3
"""HTTP MJPEG stream server that wraps GStreamer tcpserversink output."""
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

# TCP source (GStreamer tcpserversink)
TCP_HOST = 'localhost'
TCP_PORT = 8081  # GStreamer outputs here

# HTTP server port
HTTP_PORT = 8080

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8s Live - Tenstorrent</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { margin:0; background:#111; color:#fff; font-family:system-ui; }
        .container { text-align:center; padding:20px; }
        h1 { color:#00d4ff; margin-bottom:10px; }
        .stats { color:#888; margin-bottom:20px; }
        img { border:3px solid #333; border-radius:8px; display:block; margin:0 auto; }
        #status { color:#00d4ff; font-size:18px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOv8s Object Detection</h1>
        <p class="stats">Running on Tenstorrent Hardware | ~100 FPS inference</p>
        <p id="status">Loading stream... (auto-refresh every 5s)</p>
        <img id="stream" src="/stream" width="640" height="640" alt="Live Stream"
             onload="document.getElementById('status').style.display='none'; document.querySelector('meta[http-equiv=refresh]').remove();"
             onerror="document.getElementById('status').innerHTML='Model compiling... (~2 min)';">
    </div>
</body>
</html>''')
        
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            try:
                # Connect to GStreamer TCP output
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((TCP_HOST, TCP_PORT))
                sock.settimeout(5.0)
                
                # Forward the MJPEG stream
                while True:
                    data = sock.recv(65536)
                    if not data:
                        break
                    self.wfile.write(data)
                    self.wfile.flush()
            except Exception as e:
                print(f"Stream error: {e}")
            finally:
                sock.close()
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Quiet

if __name__ == '__main__':
    print(f"Starting HTTP server on port {HTTP_PORT}")
    print(f"Forwarding from TCP port {TCP_PORT}")
    print(f"Open: http://localhost:{HTTP_PORT}")
    server = HTTPServer(('0.0.0.0', HTTP_PORT), MJPEGHandler)
    server.serve_forever()
