#!/usr/bin/env python3

"""
Example of using CleanedPromptGenerator with server-side tokenization
Requires a running inference server on localhost:8000
"""

import os
from cleaned_prompt_generator import CleanedPromptGenerator
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig

def setup_server_client():
    """Set up the client for server-side tokenization"""
    
    # Set up environment variables if needed
    # os.environ['TOKEN'] = 'your_auth_token_here'
    
    # Create environment config
    env_config = EnvironmentConfig(
            deploy_url="http://127.0.0.1",
            service_port='8000',
            authorization=None,
            jwt_secret="test1234",
            vllm_model="meta-llama/Llama-3.1-8B-Instruct",
            mesh_device="n300",
            cache_root='/home/user/tt-inference-server'
        )
    
    # Create prompt client
    client = PromptClient(env_config)
    
    # Wait for server to be healthy (optional)
    print("Checking server health...")
    if client.wait_for_healthy(timeout=30):
        print("Server is healthy and ready!")
        return client
    else:
        print("Server health check failed!")
        return None

def main():
    print("=== Server-Side Tokenization Example ===")
    
    # Set up the server client
    client = setup_server_client()
    if client is None:
        print("Failed to connect to server. Exiting.")
        return
    
    # Initialize with server-side tokenization
    generator = CleanedPromptGenerator(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        server_tokenizer=True,  # Use server-side tokenization
        client=client,          # Pass the PromptClient instance
        seed=42
    )
    
    print("Generator initialized with server-side tokenization")
    
    try:
        # Generate stable tokens using server
        tokens = generator.generate_stable_tokens(
            input_length=256,
            max_length=512
        )
        
        print(f"Generated {len(tokens)} stable tokens using server tokenization")
        print(f"Sample tokens: {tokens[:20]}")
        
        # Generate multiple sequences
        sequences = generator.generate_multiple_stable_tokens(
            input_length=128,
            max_length=256,
            num_sequences=2,
            base_seed=200
        )
        
        print(f"\nGenerated {len(sequences)} sequences using server:")
        for i, seq in enumerate(sequences):
            print(f"  Sequence {i+1}: {len(seq)} tokens")
            
    except Exception as e:
        print(f"Error during server tokenization: {e}")
        print("Make sure the server is running and accessible")

if __name__ == "__main__":
    main() 