#!/usr/bin/env python3
"""
SpeechT5 TTS Server Stability Test

Sends 12+ sequential requests to the SpeechT5 server to verify stability.
Tests various text lengths and monitors for hangs, timeouts, or errors.

Usage:
    python tests/test_speecht5_stability.py [--socket /tmp/tts_server.sock] [--requests 12]
"""

import os
import sys
import json
import socket
import time
import argparse
from typing import Tuple, Optional


TTS_SOCKET = "/tmp/tts_server.sock"

TEST_PHRASES = [
    "Hello, how are you today?",
    "Welcome to the voice assistant demo.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing one two three four five.",
    "This is a longer sentence to test how the system handles more complex text with multiple words and phrases.",
    "Short test.",
    "The weather today is sunny with a high of seventy five degrees.",
    "Please remember to save your work before closing the application.",
    "Artificial intelligence is transforming how we interact with technology.",
    "Thank you for using our voice assistant service.",
    "Let me help you with that question.",
    "Here is the information you requested about the meeting schedule.",
    "The system is processing your request. Please wait a moment.",
    "Your account has been updated successfully.",
    "Would you like me to read that again?",
    "I'm sorry, I didn't understand. Could you please repeat that?",
]


def send_request(socket_path: str, request: dict, timeout: float = 120) -> Tuple[dict, float]:
    """Send request to TTS server and return response with elapsed time."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    
    start_time = time.time()
    try:
        client.connect(socket_path)
        client.sendall(json.dumps(request).encode('utf-8'))
        response = client.recv(65536).decode('utf-8')
        elapsed = time.time() - start_time
        return json.loads(response), elapsed
    finally:
        client.close()


def check_server_ready(socket_path: str) -> bool:
    """Check if TTS server is ready."""
    try:
        response, _ = send_request(socket_path, {"cmd": "ping"}, timeout=10)
        return response.get("status") == "ok"
    except Exception as e:
        print(f"Server not ready: {e}")
        return False


def run_stability_test(socket_path: str, num_requests: int, verbose: bool = True) -> dict:
    """Run stability test with multiple sequential requests."""
    
    print("=" * 70)
    print(f"SpeechT5 TTS Stability Test")
    print(f"Socket: {socket_path}")
    print(f"Requests: {num_requests}")
    print("=" * 70)
    
    # Check server is ready
    print("\n[1] Checking server is ready...")
    if not check_server_ready(socket_path):
        print("ERROR: TTS server is not running or not responding!")
        print(f"Make sure the server is started and listening on {socket_path}")
        return {"success": False, "error": "Server not ready"}
    print("Server is ready!\n")
    
    # Run tests
    results = {
        "total_requests": num_requests,
        "successful": 0,
        "failed": 0,
        "timeouts": 0,
        "errors": [],
        "timings": [],
        "avg_time_ms": 0,
        "min_time_ms": float('inf'),
        "max_time_ms": 0,
    }
    
    print("[2] Running stability test...")
    print("-" * 70)
    
    output_dir = "/tmp/tts_stability_test"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_requests):
        phrase = TEST_PHRASES[i % len(TEST_PHRASES)]
        output_path = f"{output_dir}/test_{i+1:03d}.wav"
        
        if verbose:
            print(f"\nRequest {i+1}/{num_requests}:")
            print(f"  Text: '{phrase[:50]}{'...' if len(phrase) > 50 else ''}'")
        
        try:
            response, elapsed_sec = send_request(
                socket_path,
                {"text": phrase, "output_path": output_path},
                timeout=120
            )
            
            elapsed_ms = elapsed_sec * 1000
            
            if response.get("status") == "ok":
                results["successful"] += 1
                results["timings"].append(elapsed_ms)
                results["min_time_ms"] = min(results["min_time_ms"], elapsed_ms)
                results["max_time_ms"] = max(results["max_time_ms"], elapsed_ms)
                
                if verbose:
                    server_time = response.get("time_ms", 0)
                    print(f"  Status: OK")
                    print(f"  Server time: {server_time:.0f}ms")
                    print(f"  Round-trip: {elapsed_ms:.0f}ms")
                    
                    # Check output file
                    if os.path.exists(output_path):
                        size = os.path.getsize(output_path)
                        print(f"  Output: {output_path} ({size} bytes)")
                    else:
                        print(f"  WARNING: Output file not created!")
            else:
                results["failed"] += 1
                error_msg = response.get("error", "Unknown error")
                results["errors"].append(f"Request {i+1}: {error_msg}")
                if verbose:
                    print(f"  Status: FAILED - {error_msg}")
                    
        except socket.timeout:
            results["timeouts"] += 1
            results["errors"].append(f"Request {i+1}: TIMEOUT (120s)")
            if verbose:
                print(f"  Status: TIMEOUT")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Request {i+1}: {str(e)}")
            if verbose:
                print(f"  Status: ERROR - {e}")
        
        # Brief pause between requests to mimic real usage
        time.sleep(0.5)
    
    # Calculate statistics
    if results["timings"]:
        results["avg_time_ms"] = sum(results["timings"]) / len(results["timings"])
    
    if results["min_time_ms"] == float('inf'):
        results["min_time_ms"] = 0
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Requests:    {results['total_requests']}")
    print(f"Successful:        {results['successful']} ({100*results['successful']/results['total_requests']:.1f}%)")
    print(f"Failed:            {results['failed']}")
    print(f"Timeouts:          {results['timeouts']}")
    
    if results["timings"]:
        print(f"\nTiming Statistics:")
        print(f"  Average:         {results['avg_time_ms']:.0f}ms")
        print(f"  Min:             {results['min_time_ms']:.0f}ms")
        print(f"  Max:             {results['max_time_ms']:.0f}ms")
    
    if results["errors"]:
        print(f"\nErrors:")
        for error in results["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    # Determine overall success
    success_rate = results["successful"] / results["total_requests"]
    if success_rate >= 0.95:
        print(f"\n✅ TEST PASSED - {success_rate*100:.0f}% success rate")
        results["success"] = True
    else:
        print(f"\n❌ TEST FAILED - {success_rate*100:.0f}% success rate")
        results["success"] = False
    
    return results


def run_stress_test(socket_path: str, duration_minutes: int = 5) -> dict:
    """Run continuous stress test for specified duration."""
    print("=" * 70)
    print(f"SpeechT5 TTS Stress Test ({duration_minutes} minutes)")
    print("=" * 70)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    request_count = 0
    successful = 0
    failed = 0
    
    while time.time() < end_time:
        request_count += 1
        phrase = TEST_PHRASES[request_count % len(TEST_PHRASES)]
        
        try:
            response, _ = send_request(
                socket_path,
                {"text": phrase, "output_path": f"/tmp/stress_{request_count}.wav"},
                timeout=60
            )
            if response.get("status") == "ok":
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"  Request {request_count} failed: {e}")
        
        elapsed = time.time() - start_time
        print(f"\r  Progress: {elapsed/60:.1f}/{duration_minutes} min | "
              f"Requests: {request_count} | Success: {successful} | Failed: {failed}", end="")
        
        time.sleep(0.5)
    
    print(f"\n\nStress test complete:")
    print(f"  Total requests: {request_count}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    return {
        "duration_minutes": duration_minutes,
        "total_requests": request_count,
        "successful": successful,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(description="SpeechT5 TTS Server Stability Test")
    parser.add_argument("--socket", default=TTS_SOCKET, help="TTS server socket path")
    parser.add_argument("--requests", type=int, default=12, help="Number of test requests")
    parser.add_argument("--stress", type=int, default=0, help="Run stress test for N minutes (0=disabled)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()
    
    if args.stress > 0:
        results = run_stress_test(args.socket, args.stress)
    else:
        results = run_stability_test(args.socket, args.requests, verbose=not args.quiet)
    
    # Exit with appropriate code
    sys.exit(0 if results.get("success", True) else 1)


if __name__ == "__main__":
    main()
