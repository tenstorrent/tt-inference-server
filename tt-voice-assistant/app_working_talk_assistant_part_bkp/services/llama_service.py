"""
Llama Chat Service - Direct TT Metal Inference
Based on working llama_inference_clean.py
Device: 1 (from NOTES.md)
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LlamaService:
    """Llama 3.1 8B Instruct on TT Metal Device 1."""
    
    def __init__(self, device_id: int = 1):
        """Initialize Llama service."""
        self.device_id = device_id
        self.service_name = "Llama"
        self.is_warmed_up = False
        self.warmup_time = 0
        
        # Model components (loaded during warmup)
        self.mesh_device = None
        self.model_args = None
        self.generator = None
        self.tt_kv_cache = None
        self.tokenizer = None
        
        logger.info(f"Llama service initialized for device {device_id}")
    
    def format_prompt(self, user_message: str) -> str:
        """Format prompt for Llama 3.1 Instruct."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful voice assistant. Give brief, direct answers suitable for speech.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    async def warmup(self):
        """Load and warm up Llama model on TT Metal."""
        logger.info(f"🔥 Warming up Llama 3.1 8B on device {self.device_id}...")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Set environment for this device
            os.environ['HF_MODEL'] = 'meta-llama/Llama-3.1-8B-Instruct'
            os.environ['TT_MESH_GRAPH_DESC_PATH'] = '/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto'
            os.environ['TT_VISIBLE_DEVICES'] = str(self.device_id)
            
            # Add TT Metal to path
            sys.path.insert(0, "/home/container_app_user/tt-metal")
            
            # Change to tt-metal directory so model_cache path is found
            os.chdir("/home/container_app_user/tt-metal")
            
            # Import TT Metal modules
            import torch
            import ttnn
            from models.tt_transformers.tt.common import create_tt_model
            from models.tt_transformers.tt.generator import Generator
            from models.tt_transformers.tt.model_config import DecodersPrecision
            
            # Create mesh device for this specific device
            logger.info(f"Opening mesh device on physical device {self.device_id}...")
            self.mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
            
            # Create model with performance optimizations
            logger.info("Loading Llama model...")
            optimizations = lambda m: DecodersPrecision.performance(m.n_layers, m.model_name)
            self.model_args, model, self.tt_kv_cache, _ = create_tt_model(
                self.mesh_device,
                instruct=True,
                max_batch_size=1,
                optimizations=optimizations,
                max_seq_len=512,
                dtype=ttnn.bfloat8_b,
            )
            
            # Create generator
            logger.info("Creating generator...")
            self.generator = Generator(
                [model], [self.model_args], self.mesh_device,
                processor=self.model_args.processor,
                tokenizer=self.model_args.tokenizer
            )
            
            self.tokenizer = self.model_args.tokenizer
            
            # Run warmup inference
            logger.info("Running warmup inference...")
            _ = await self.generate_response("Hello")
            
            self.warmup_time = asyncio.get_event_loop().time() - start_time
            self.is_warmed_up = True
            logger.info(f"✅ Llama warmed up in {self.warmup_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Llama warmup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def generate_response(self, message: str, max_tokens: int = 256) -> Dict[str, Any]:
        """Generate response using Llama."""
        logger.info(f"🤖 Generating response to: {message[:50]}...")
        
        try:
            import torch
            start_time = asyncio.get_event_loop().time()
            
            # Tokenize
            formatted = self.format_prompt(message)
            tokens = self.tokenizer.encode(formatted)
            input_tokens = torch.tensor([tokens])
            
            # Prefill
            result = self.generator.prefill_forward_text(
                input_tokens,
                page_table=None,
                kv_cache=self.tt_kv_cache,
                prompt_lens=[len(tokens)],
            )
            logits = result[0] if isinstance(result, tuple) else result
            
            # Decode loop
            generated_tokens = list(tokens)
            
            for _ in range(max_tokens):
                next_token_id = torch.argmax(logits[0, -1, :]).item()
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token_id)
                
                result = self.generator.decode_forward(
                    torch.tensor([[next_token_id]]),
                    torch.tensor([len(generated_tokens)])
                )
                logits = result[0] if isinstance(result, tuple) else result
            
            response = self.tokenizer.decode(
                generated_tokens[len(tokens):], 
                skip_special_tokens=True
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"✅ Response generated in {processing_time:.2f}s")
            
            return {
                "response": response.strip(),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {
                "response": "I'm sorry, I encountered an error.",
                "processing_time": 0
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time
        }
    
    async def shutdown(self):
        """Shutdown Llama service."""
        logger.info("🛑 Shutting down Llama service...")
        try:
            if self.mesh_device:
                import ttnn
                ttnn.close_mesh_device(self.mesh_device)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        self.is_warmed_up = False