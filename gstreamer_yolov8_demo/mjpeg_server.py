#!/usr/bin/env python3
"""Simple HTTP MJPEG server that wraps GStreamer output."""
import sys
import time
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

# Global frame buffer
current_frame = None
frame_lock = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
            <html><body style="margin:0;background:#000;">
            <h2 style="color:white;text-align:center;">YOLOv8s on Tenstorrent - Live Stream</h2>
            <img src="/stream" style="display:block;margin:auto;max-width:100%;">
            </body></html>
            ''')
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            import os
            frame_dir = '/workspace'
            frame_num = 0
            
            while True:
                # Read latest frame file
                frame_path = f'{frame_dir}/live_frame_{frame_num:03d}.jpg'
                try:
                    if os.path.exists(frame_path):
                        with open(frame_path, 'rb') as f:
                            frame_data = f.read()
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                        frame_num = (frame_num + 1) % 1000
                except:
                    pass
                time.sleep(0.033)  # ~30fps
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == '__main__':
    port = 8080
    print(f"Starting MJPEG server on port {port}...")
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    server.serve_forever()
