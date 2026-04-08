#!/usr/bin/env python3
"""
Register a face with the Face Auth server.

Usage:
  python register_face.py --name "Teja" --image /path/to/face.jpg
  
Or capture from webcam (if available):
  python register_face.py --name "Teja" --webcam
"""

import socket
import json
import base64
import argparse
import sys

SOCKET_PATH = "/tmp/face_auth_server.sock"

def register_face(name: str, image_path: str = None, image_data: bytes = None):
    """Register a face with the server."""
    if image_path:
        with open(image_path, "rb") as f:
            image_data = f.read()
    
    if not image_data:
        print("Error: No image provided")
        return False
    
    image_b64 = base64.b64encode(image_data).decode('utf-8')
    
    request = {
        "cmd": "register",
        "name": name,
        "image_base64": image_b64
    }
    
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30)
    client.connect(SOCKET_PATH)
    client.sendall(json.dumps(request).encode('utf-8'))
    response = client.recv(65536).decode('utf-8')
    client.close()
    
    result = json.loads(response)
    print(f"Response: {result}")
    
    if result.get("status") == "ok":
        print(f"✅ Successfully registered: {result.get('registered')}")
        return True
    else:
        print(f"❌ Registration failed: {result.get('error')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Register a face")
    parser.add_argument("--name", required=True, help="Name to register")
    parser.add_argument("--image", help="Path to face image (jpg/png)")
    args = parser.parse_args()
    
    if args.image:
        register_face(args.name, image_path=args.image)
    else:
        print("Please provide --image /path/to/face.jpg")
        sys.exit(1)


if __name__ == "__main__":
    main()
