# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.settings import settings
from loguru import logger
import time
import torch
from tqdm import tqdm
import ttnn
from typing import List
from tests.scripts.common import get_updated_device_params
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger
import numpy as np

from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache
from models.generation_utils import get_logits_processor

class TTWhisperRunner(DeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()
        self.device = None
        self.pipeline = None
        self.ttnn_model = None

    def _set_fabric(self,fabric_config):
        # If fabric_config is not None, set it to fabric_config
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        # for now use all available devices
        return self._mesh_device()

    def _mesh_device(self):
        device_params = {'l1_small_size': WHISPER_L1_SMALL_SIZE}
        device_ids = ttnn.get_device_ids()

        param = len(device_ids)  # Default to using all available devices

        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if num_devices_requested > len(device_ids):
                print("Requested more devices than available. Test not applicable for machine")
            mesh_shape = ttnn.MeshShape(*grid_dims)
            assert num_devices_requested <= len(device_ids), "Requested more devices than available."
        else:
            num_devices_requested = min(param, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)


        updated_device_params = get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def get_devices(self) -> List[ttnn.MeshDevice]:
        device = self._mesh_device()
        device_shape = settings.device_mesh_shape
        return (device, device.create_submeshes(ttnn.MeshShape(*device_shape)))

    def close_device(self, device) -> bool:
        if device is None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
        else:
            ttnn.close_mesh_device(device)
        return True

    async def load_model(self, device)->bool:
        self.logger.info("Loading model...")
        if device is None:
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device

        # Prepare the inference pipeline
        self.ttnn_model = ttnn_optimized_functional_whisper
        self.pipeline = self._create_functional_whisper_for_conditional_generation_inference_pipeline()

        self.logger.info("Whisper model loaded and pipeline ready")


        # Warmup: run a dummy inference with a short silent audio
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
        try:
            _ = self.pipeline(dummy_audio, 16000, stream=False)
            self.logger.info("Model warmup completed")
        except Exception as e:
            self.logger.error(f"Model warmup failed: {e}")

        return True

    def run_inference(self, audio_data, sampling_rate, stream=False, return_perf_metrics=False):
        if self.pipeline is None:
            raise RuntimeError("Model pipeline not loaded. Call load_model() first.")
        self.logger.info(f"Running inference on audio data, duration: {len(audio_data)/sampling_rate:.2f}s")
        return self.pipeline(audio_data, sampling_rate, stream=stream, return_perf_metrics=return_perf_metrics)

    def _load_conditional_generation_ref_model():
        hf_ref_model = (
            WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3").to(torch.bfloat16).eval()
        )
        processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v3", language="English", task="transcribe")
        feature_extractor = AutoFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3")
        config = hf_ref_model.config
        return (
            hf_ref_model,
            config,
            processor,
            feature_extractor,
        )
    
    def _init_conditional_generation_tt_model(self, hf_ref_model, config, device, max_batch_size=1, max_seq_len=512):
        model = hf_ref_model.model
        linear_weight = hf_ref_model.proj_out.weight

        ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
        ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            convert_to_ttnn=self.ttnn_model.convert_to_ttnn,
            custom_preprocessor=self.ttnn_model.custom_preprocessor,
            device=device,
        )

        # Note: config.max_length is 448 for distil-whisper/distil-large-v3
        kv_cache = init_kv_cache(config, device, max_batch_size, max_seq_len=max_seq_len)

        return parameters, ttnn_linear_weight, kv_cache
    
    def _run_generate(
        self,
        config,
        audio_data,
        sampling_rate,
        feature_extractor,
        parameters,
        processor,
        ttnn_linear_weight,
        device,
        generation_config,
        kv_cache=None,
        stream_generation=False,
        feature_dtype_to_use=torch.bfloat16,
        return_perf_metrics=False,
    ):
        start_encode = time.time()

        # Compute features
        inputs = feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.type(feature_dtype_to_use)
        unpadded_batch_size = input_features.shape[0]
        assert unpadded_batch_size == 1, "Only batch size 1 is supported for inference"

        # Compute embeddings
        input_embeds = self.ttnn_model.preprocess_encoder_inputs(
            config, input_features, parameters=parameters.encoder, device=device
        )

        # Run encoder
        encoder_hidden_states = self.ttnn_model.encoder(config, input_embeds, parameters=parameters.encoder)
        ttnn.synchronize_device(device)
        logger.info(f"Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

        # Run decoder

        def _run_generate():
            def pad_input_32(tensor, value):
                len = tensor.shape[1]

                if len % 32 == 0:
                    return tensor

                padded_len = ((len // 32) + 1) * 32

                pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
                tensor = torch.cat([tensor, pad_tensor], dim=1)

                return tensor
    
            # Input ids
            input_ids = torch.tensor([[1]]) * config.decoder_start_token_id
            logits_processor = get_logits_processor(input_ids, config)
            if not kv_cache:
                input_ids = pad_input_32(input_ids, config.pad_token_id).to(torch.long)
                decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

            # Initial decode position
            current_decode_pos = (
                ttnn.from_torch(torch.zeros(unpadded_batch_size), device=device, dtype=ttnn.int32) if kv_cache else None
            )

            MAX_GEN_LEN = config.max_length  # 448 for distil-whisper/distil-large-v3
            print_each_iter = False
            output_ids = []
            total_decode_time = 0
            for i in tqdm(range(MAX_GEN_LEN), desc="Decode inference iterations"):
                start_iter = time.time()

                decoder_hidden_states, decoder_attention_mask = self.ttnn_model.preprocess_decoder_inputs(
                    config=config,
                    input_ids=input_ids,
                    attention_mask=None,
                    parameters=parameters.decoder,
                    device=device,
                    decode_pos=i if kv_cache else None,
                    create_attention_mask=(not kv_cache),
                )

                output = self.ttnn_model.decoder(
                    config,
                    decoder_hidden_states,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    kv_cache=kv_cache,
                    current_decode_pos=current_decode_pos,
                    parameters=parameters.decoder,
                )

                if not kv_cache:
                    # Note: if not using a kv cache, the entire sequence is recomputed at each step
                    # Only run the lm head on the last tile to fix bad outputs and reduce redundant computation
                    last_tile_start_idx = i // 32 * 32
                    output_idx = i % 32
                    output = output[:, last_tile_start_idx : last_tile_start_idx + 32, :]
                else:
                    output_idx = 0

                output = output @ ttnn_linear_weight
                logits_to_torch = ttnn.to_torch(output)
                next_token_logits = logits_to_torch[:, output_idx, :]

                next_tokens_scores = logits_processor(input_features, next_token_logits)
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                output_ids.append(next_tokens)

                if i == 0:
                    first_token_time = time.time()
                    ttft = first_token_time - start_encode

                # Update input_ids and current_decode_pos
                if not kv_cache:
                    if (i + 1) % 32 == 0:
                        input_ids = torch.cat([input_ids, decoder_start_values], dim=1)
                    input_ids[:, i + 1] = next_tokens[:, None]
                else:
                    input_ids = next_tokens[:, None]
                    ttnn.plus_one(current_decode_pos)

                total_decode_time += time.time() - start_iter
                avg_decode_throughput = (i + 1) / total_decode_time

                ttnn_transcription = processor.batch_decode(next_tokens.unsqueeze(dim=1), skip_special_tokens=True)[0]
                if print_each_iter:
                    logger.info(processor.batch_decode(torch.stack(output_ids, dim=1), skip_special_tokens=True)[0])

                if return_perf_metrics:
                    yield ttnn_transcription, ttft, avg_decode_throughput
                else:
                    yield ttnn_transcription

                if next_tokens == config.eos_token_id:
                    break

            total_generate_time = time.time() - start_encode
            logger.info(f"Time to first token: {(ttft*1000):.3f}ms")
            logger.info(f"Total decode time: {total_decode_time:.3f}s")
            logger.info(f"Total generate time: {total_generate_time:.3f}s")
            logger.info(f"Average decode throughput: {avg_decode_throughput:.3f} t/s/u")

        # conditionally return generator or full response
        if stream_generation:
            return _run_generate()
        else:
            output = []
            for x in _run_generate():
                if return_perf_metrics:
                    out_cur, ttft, avg_decode_throughput = x
                else:
                    out_cur = x
                output.append(out_cur)
            output = "".join(output)

            if return_perf_metrics:
                return output, ttft, avg_decode_throughput
            else:
                return output

    def _create_functional_whisper_for_conditional_generation_inference_pipeline(self):
        """
        Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
        and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
        signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
        the callable returns the full decoded output.
        """
        hf_ref_model, config, processor, feature_extractor = self._load_conditional_generation_ref_model()
        parameters, ttnn_linear_weight, kv_cache = self._init_conditional_generation_tt_model(
            hf_ref_model, config, self.device
        )

        def _model_pipeline(data, sampling_rate, stream=False, return_perf_metrics=False):
            logger.info(f"Running model on audio data with duration {data.shape[0]/sampling_rate:.3f}s")

            return self._run_generate(
                config,
                data,
                sampling_rate,
                feature_extractor,
                parameters=parameters,
                processor=processor,
                ttnn_linear_weight=ttnn_linear_weight,
                device=self.device,
                generation_config=hf_ref_model.generation_config,
                kv_cache=kv_cache,
                stream_generation=stream,
                return_perf_metrics=return_perf_metrics,
            )

        return _model_pipeline
