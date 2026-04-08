#!/usr/bin/env python3
"""
Simple Qwen3 TTS test - no trace, just verify model loads and runs on Blackhole.

Usage:
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='3' \
  python test_qwen3_tts_simple.py
"""

import os
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='Hello from Tenstorrent Blackhole!')
    args = parser.parse_args()
    
    print('='*60)
    print('Qwen3 TTS Simple Test (No Trace)')
    print('='*60)
    
    import torch
    import torch.nn.functional as F
    import ttnn
    import soundfile as sf
    from pathlib import Path
    
    # Open device
    print('\nOpening device...')
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    device.enable_program_cache()
    print(f'Device opened: {device}')
    
    try:
        # Load weights
        print('\nLoading model weights...')
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        
        model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
        
        main_dict = {}
        for f in model_path.glob("*.safetensors"):
            main_dict.update(load_file(f))
        main_dict = {k: v.float() for k, v in main_dict.items()}
        
        speech_path = model_path / "speech_tokenizer" / "model.safetensors"
        speech_dict = load_file(speech_path)
        decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}
        
        print(f'  Loaded {len(main_dict)} main weights')
        print(f'  Loaded {len(decoder_weights)} decoder weights')
        
        # Initialize full TTNN model
        print('\nInitializing TTNN Qwen3TTS model...')
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
        from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat
        
        t_init_start = time.time()
        model = Qwen3TTS(
            device=device,
            state_dict=main_dict,
        )
        t_init_end = time.time()
        print(f'  Model initialized in {(t_init_end - t_init_start):.1f}s')
        
        # Test speaker encoder with reference audio
        print('\nTesting speaker encoder...')
        ref_audio_path = Path('/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav')
        if ref_audio_path.exists():
            audio_data, sr = sf.read(ref_audio_path)
            audio_tensor = torch.from_numpy(audio_data).float()
            print(f'  Loaded reference audio: {len(audio_data)/sr:.2f}s @ {sr}Hz')
        else:
            audio_tensor = torch.randn(24000)
            print('  Using dummy audio (reference not found)')
        
        t0 = time.time()
        spk_embed = model.extract_speaker_embedding(audio_tensor)
        t1 = time.time()
        print(f'  Speaker embedding shape: {spk_embed.shape}')
        print(f'  Time: {(t1-t0)*1000:.1f}ms')
        
        # Test Talker forward (single step, no KV cache, no trace)
        print('\nTesting Talker forward (single step)...')
        
        # Get transformation matrix and RoPE
        trans_mat = get_transformation_mat(model.talker_config.head_dim, device)
        position_ids = torch.arange(1).unsqueeze(0)  # [1, seq_len]
        cos, sin = get_rope_tensors(device, model.talker_config.head_dim, 1, position_ids, model.talker_config.rope_theta)
        
        # Create dummy input embedding
        dummy_embed = torch.randn(1, 1, 1, model.talker_config.hidden_size, dtype=torch.bfloat16)
        dummy_embed_tt = ttnn.from_torch(
            dummy_embed,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        t0 = time.time()
        hidden_out, _ = model.talker.forward_from_hidden(
            dummy_embed_tt,
            cos,
            sin,
            trans_mat,
            kv_caches=None,  # No KV cache
            mode="prefill",
        )
        hidden_torch = ttnn.to_torch(hidden_out)
        t1 = time.time()
        print(f'  Talker output shape: {hidden_torch.shape}')
        print(f'  Time: {(t1-t0)*1000:.1f}ms')
        
        # Test codec logits
        print('\nTesting codec head...')
        t0 = time.time()
        logits = model.talker.get_codec_logits(hidden_out)
        logits_torch = ttnn.to_torch(logits)
        t1 = time.time()
        print(f'  Codec logits shape: {logits_torch.shape}')
        print(f'  Time: {(t1-t0)*1000:.1f}ms')
        
        print('\n' + '='*60)
        print('SUCCESS: Qwen3 TTS core components work on Blackhole!')
        print('='*60)
        print('\nNote: Full generation requires trace-free decode loop.')
        print('The trace capture issue on Blackhole needs to be fixed in attention.py')
        
    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print('\nClosing device...')
        try:
            ttnn.close_device(device)
        except:
            pass
        print('Done.')

if __name__ == '__main__':
    main()
