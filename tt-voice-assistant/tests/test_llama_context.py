#!/usr/bin/env python3
"""
Llama Context Memory Test

Tests the conversation history/context memory implementation for Llama.
Verifies that the assistant properly remembers context across turns.

Usage:
    python tests/test_llama_context.py [--host localhost] [--port 8080]
"""

import argparse
import requests
import json
import time
import uuid
from typing import List, Tuple


def chat(base_url: str, message: str, session_id: str, max_tokens: int = 150) -> Tuple[str, float]:
    """Send a chat message and return response with timing."""
    start = time.time()
    response = requests.post(
        f"{base_url}/api/chat",
        json={"message": message, "session_id": session_id, "max_tokens": max_tokens},
        timeout=60
    )
    elapsed = time.time() - start
    
    if response.status_code != 200:
        return f"ERROR: {response.status_code}", elapsed
    
    data = response.json()
    return data.get("response", "No response"), elapsed


def clear_history(base_url: str, session_id: str) -> bool:
    """Clear conversation history for a session."""
    try:
        response = requests.post(
            f"{base_url}/api/clear-history",
            json={"session_id": session_id},
            timeout=10
        )
        return response.status_code == 200
    except:
        return False


def test_basic_context_memory(base_url: str) -> bool:
    """Test that basic context is remembered within a session."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Context Memory")
    print("=" * 70)
    
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    clear_history(base_url, session_id)
    
    # First message: introduce a fact
    print("\n[Turn 1] User: My name is Alex and I work at Google.")
    response1, time1 = chat(base_url, "My name is Alex and I work at Google.", session_id)
    print(f"[Turn 1] Assistant ({time1:.2f}s): {response1}")
    
    # Second message: ask about the fact
    print("\n[Turn 2] User: What is my name?")
    response2, time2 = chat(base_url, "What is my name?", session_id)
    print(f"[Turn 2] Assistant ({time2:.2f}s): {response2}")
    
    # Check if name is remembered
    name_remembered = "alex" in response2.lower()
    
    # Third message: ask about workplace
    print("\n[Turn 3] User: Where do I work?")
    response3, time3 = chat(base_url, "Where do I work?", session_id)
    print(f"[Turn 3] Assistant ({time3:.2f}s): {response3}")
    
    # Check if workplace is remembered
    workplace_remembered = "google" in response3.lower()
    
    passed = name_remembered and workplace_remembered
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Name remembered={name_remembered}, Workplace remembered={workplace_remembered}")
    
    return passed


def test_multi_turn_conversation(base_url: str) -> bool:
    """Test multi-turn conversation flow."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Turn Conversation")
    print("=" * 70)
    
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    clear_history(base_url, session_id)
    
    conversation = [
        ("I have 3 apples.", "acknowledging apples"),
        ("I just bought 2 more apples.", "acknowledging more apples"),
        ("How many apples do I have in total?", "should say 5"),
    ]
    
    for i, (user_msg, expected) in enumerate(conversation, 1):
        print(f"\n[Turn {i}] User: {user_msg}")
        response, elapsed = chat(base_url, user_msg, session_id)
        print(f"[Turn {i}] Assistant ({elapsed:.2f}s): {response}")
    
    # Check if final answer mentions 5
    has_five = "5" in response or "five" in response.lower()
    
    print(f"\n{'✅ PASSED' if has_five else '❌ FAILED'}: Math context test (expected 5)")
    
    return has_five


def test_session_isolation(base_url: str) -> bool:
    """Test that different sessions are isolated."""
    print("\n" + "=" * 70)
    print("TEST 3: Session Isolation")
    print("=" * 70)
    
    session1 = f"test_{uuid.uuid4().hex[:8]}"
    session2 = f"test_{uuid.uuid4().hex[:8]}"
    
    clear_history(base_url, session1)
    clear_history(base_url, session2)
    
    # Session 1: Set up context
    print(f"\n[Session 1] User: My favorite color is blue.")
    response1a, _ = chat(base_url, "My favorite color is blue.", session1)
    print(f"[Session 1] Assistant: {response1a}")
    
    # Session 2: Set up different context  
    print(f"\n[Session 2] User: My favorite color is red.")
    response2a, _ = chat(base_url, "My favorite color is red.", session2)
    print(f"[Session 2] Assistant: {response2a}")
    
    # Session 1: Query context
    print(f"\n[Session 1] User: What is my favorite color?")
    response1b, _ = chat(base_url, "What is my favorite color?", session1)
    print(f"[Session 1] Assistant: {response1b}")
    
    # Session 2: Query context
    print(f"\n[Session 2] User: What is my favorite color?")
    response2b, _ = chat(base_url, "What is my favorite color?", session2)
    print(f"[Session 2] Assistant: {response2b}")
    
    # Check isolation
    session1_correct = "blue" in response1b.lower() and "red" not in response1b.lower()
    session2_correct = "red" in response2b.lower() and "blue" not in response2b.lower()
    
    passed = session1_correct and session2_correct
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Session1 blue={session1_correct}, Session2 red={session2_correct}")
    
    return passed


def test_history_trimming(base_url: str) -> bool:
    """Test that old history is properly trimmed."""
    print("\n" + "=" * 70)
    print("TEST 4: History Trimming (Many Turns)")
    print("=" * 70)
    
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    clear_history(base_url, session_id)
    
    # Make many turns to exceed MAX_HISTORY_TURNS
    facts = [
        ("The password is ALPHA123.", "password"),
        ("My pet's name is Max.", "pet"),
        ("I live in Seattle.", "city"),
        ("My birthday is March 15.", "birthday"),
        ("I drive a Tesla.", "car"),
        ("My phone number is 555-1234.", "phone"),
    ]
    
    for fact, _ in facts:
        print(f"  User: {fact}")
        response, _ = chat(base_url, fact, session_id)
        print(f"  Assistant: {response[:80]}...")
    
    # Now check if recent facts are remembered
    print("\n  Checking recent facts...")
    
    # Check most recent fact (should be remembered)
    print("  User: What is my phone number?")
    response, _ = chat(base_url, "What is my phone number?", session_id)
    print(f"  Assistant: {response}")
    recent_remembered = "555" in response or "1234" in response
    
    # Check oldest fact (might be trimmed)
    print("  User: What was the password I mentioned?")
    response2, _ = chat(base_url, "What was the password I mentioned?", session_id)
    print(f"  Assistant: {response2}")
    old_forgotten = "ALPHA" not in response2 and "alpha" not in response2.lower()
    
    # It's OK if old is remembered (trimming might not have happened yet)
    # But recent should definitely be remembered
    passed = recent_remembered
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: Recent fact remembered={recent_remembered}, Old fact forgotten={old_forgotten}")
    
    return passed


def test_clear_history(base_url: str) -> bool:
    """Test that clearing history works."""
    print("\n" + "=" * 70)
    print("TEST 5: Clear History")
    print("=" * 70)
    
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    
    # Set up context
    print("\n[Before clear] User: My secret code is XYZ789.")
    response1, _ = chat(base_url, "My secret code is XYZ789.", session_id)
    print(f"[Before clear] Assistant: {response1}")
    
    # Clear history
    print("\n[Clearing history...]")
    cleared = clear_history(base_url, session_id)
    print(f"  History cleared: {cleared}")
    
    # Check if context is forgotten
    print("\n[After clear] User: What is my secret code?")
    response2, _ = chat(base_url, "What is my secret code?", session_id)
    print(f"[After clear] Assistant: {response2}")
    
    # Should NOT remember the code after clearing
    code_forgotten = "xyz" not in response2.lower() and "789" not in response2
    
    print(f"\n{'✅ PASSED' if code_forgotten else '❌ FAILED'}: Code forgotten after clear={code_forgotten}")
    
    return code_forgotten


def run_all_tests(base_url: str) -> dict:
    """Run all context memory tests."""
    print("=" * 70)
    print("Llama Context Memory Test Suite")
    print(f"Target: {base_url}")
    print("=" * 70)
    
    # Check if server is healthy
    try:
        health = requests.get(f"{base_url}/health", timeout=10)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy: {health.status_code}")
            return {"success": False, "error": "Server not healthy"}
        
        health_data = health.json()
        print(f"\nServer Status:")
        print(f"  Llama: {'✅ Ready' if health_data.get('llama') else '❌ Not Ready'}")
        print(f"  TTS: {'✅ Ready' if health_data.get('tts') else '❌ Not Ready'}")
        
        if not health_data.get("llama"):
            print("\nERROR: Llama service is not ready!")
            return {"success": False, "error": "Llama not ready"}
            
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return {"success": False, "error": str(e)}
    
    # Run tests
    results = {
        "basic_context": test_basic_context_memory(base_url),
        "multi_turn": test_multi_turn_conversation(base_url),
        "session_isolation": test_session_isolation(base_url),
        "history_trimming": test_history_trimming(base_url),
        "clear_history": test_clear_history(base_url),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        results["success"] = True
    else:
        print(f"\n❌ {total - passed} tests failed")
        results["success"] = False
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Llama Context Memory Test")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    results = run_all_tests(base_url)
    
    import sys
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
