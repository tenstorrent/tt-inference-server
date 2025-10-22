#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Simple test script for tt-inference-server
"""

import requests
import jwt
import json
import sys
from pathlib import Path

def load_jwt_secret():
    """Load JWT_SECRET from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("JWT_SECRET="):
                    return line.strip().split("=", 1)[1]
    return "tenstorrent"  # fallback

def generate_token(jwt_secret):
    """Generate JWT token for authentication"""
    payload = {"team_id": "tenstorrent", "token_id": "debug-test"}
    return jwt.encode(payload, jwt_secret, algorithm="HS256")

def test_completion(prompt, max_tokens=100, temperature=0.7, url="http://localhost:8000"):
    """Send a completion request to the server"""
    jwt_secret = load_jwt_secret()
    token = generate_token(jwt_secret)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    data = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    print(f"🚀 Sending request to {url}/v1/completions")
    print(f"📝 Prompt: {prompt}")
    print(f"⚙️  Settings: max_tokens={max_tokens}, temperature={temperature}")
    print()
    
    try:
        response = requests.post(f"{url}/v1/completions", headers=headers, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        text = result['choices'][0]['text']
        usage = result.get('usage', {})
        
        print("="*80)
        print("✅ RESPONSE:")
        print("="*80)
        print(text)
        print("="*80)
        print()
        print("📊 Statistics:")
        print(f"   • Prompt tokens:     {usage.get('prompt_tokens', 'N/A')}")
        print(f"   • Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"   • Total tokens:      {usage.get('total_tokens', 'N/A')}")
        print(f"   • Finish reason:     {result['choices'][0].get('finish_reason', 'N/A')}")
        print()
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)

def main():
    """Run demo prompts"""
    print("="*80)
    print(" TT-Inference-Server Demo")
    print("="*80)
    print()
    
    # Test 1: Simple question
    test_completion(
        "What is the capital of France? Answer:",
        max_tokens=20,
        temperature=0.0  # deterministic
    )
    
    print("\n" + "="*80 + "\n")
    
    # Test 2: Code generation
    test_completion(
        "Write a Python function to calculate fibonacci numbers:\n\ndef fibonacci(n):",
        max_tokens=150,
        temperature=0.3
    )
    
    print("\n" + "="*80 + "\n")
    
    # Test 3: Creative
    test_completion(
        "Write a haiku about artificial intelligence:",
        max_tokens=50,
        temperature=0.8
    )
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom prompt from command line
        prompt = " ".join(sys.argv[1:])
        test_completion(prompt)
    else:
        main()

