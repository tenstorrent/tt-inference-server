# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path

import numpy as np
import torch
import ttnn
from config.constants import SupportedModels
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner


def model_location_generator(
    model_version,
    model_subdir="",
    download_if_ci_v2=False,
    ci_v2_timeout_in_s=300,
    endpoint_prefix="http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
    download_dir_suffix="model_weights",
):
    """
    Standalone model location generator that determines the appropriate file path
    for a model based on available locations.

    Checks locations in order:
    1. Internal MLPerf path (/mnt/MLPerf/tt_dnn-models/...)
    2. Falls back to model_version string (uses HuggingFace cache)

    :param model_version: The version identifier of the model to locate (e.g., "BAAI/bge-large-en-v1.5")
    :param model_subdir: Subdirectory within the model folder structure (default: empty string)
    :param download_if_ci_v2: Not used in standalone version (kept for API compatibility)
    :param ci_v2_timeout_in_s: Not used in standalone version (kept for API compatibility)
    :param endpoint_prefix: Not used in standalone version (kept for API compatibility)
    :param download_dir_suffix: Not used in standalone version (kept for API compatibility)
    :return: Path to model files (MLPerf path or model_version string for HF cache)
    """
    model_folder = Path("tt_dnn-models") / model_subdir
    internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
    has_internal_weka = internal_weka_path.exists()

    if has_internal_weka:
        return internal_weka_path
    else:
        # Return model_version string, which will use HuggingFace cache
        # (HF_HOME or ~/.cache/huggingface by default)
        return model_version


class VLLMBGELargeENRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def get_pipeline_device_params(self):
        return {
            "num_command_queues": 2,
            "trace_region_size": self.settings.trace_region_size,
        }

    async def warmup(self) -> bool:
        max_model_len = 384
        max_num_seqs = 8 * self.settings.device_mesh_shape[1]
        self.device_batch_size = max_num_seqs
        self.sequence_length = max_model_len
        self.performant_runner = BGEPerformantRunner(
            device=self.ttnn_device,
            device_batch_size=max_num_seqs,
            sequence_length=max_model_len,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat8_b,
            model_location_generator=model_location_generator,
            model_name="BAAI/bge-large-en-v1.5",
        )
        self.performant_runner._capture_bge_trace_2cqs()
        self.performant_runner.run()

        return True

    def run(self, requests: list[TextEmbeddingRequest]):
        for req in requests:
            if req.model != SupportedModels.BGE_LARGE_EN_V1_5.value:
                raise ValueError(
                    f"Model {req.model} is not supported by VLLMBGELargeENRunner"
                )

        result = self.performant_runner.run()
        ttnn.synchronize_device(self.ttnn_device)

        # Handle different return types from performant_runner
        return self._process_results(result, requests)

    def _ensure_list_format(self, embedding):
        """
        Convert embedding to a Python list format (JSON-serializable).

        Handles:
        - numpy arrays -> list
        - torch tensors -> list
        - TTNN tensors -> list
        - Already lists -> return as-is

        :param embedding: Embedding in various formats
        :return: Python list of floats
        """
        # If already a list, return as-is
        if isinstance(embedding, list):
            return embedding

        # Handle TTNN tensors
        embedding_type_str = str(type(embedding))
        is_ttnn_tensor = (
            "ttnn" in embedding_type_str
            and "Tensor" in embedding_type_str
            and not isinstance(embedding, torch.Tensor)
        )
        if is_ttnn_tensor:
            embedding = ttnn.to_torch(embedding)

        # Handle PyTorch tensors
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        # Handle numpy arrays
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()

        # Try to convert to list if it's iterable
        try:
            return list(embedding)
        except (TypeError, ValueError):
            # If all else fails, return as-is (will cause error but at least we tried)
            return embedding

    def _process_results(
        self,
        result,
        requests: list[TextEmbeddingRequest],
    ):
        """
        Process the result from performant_runner.run() and convert to EmbeddingResponse.

        Handles different return types:
        - List of objects with .outputs.embedding and .prompt_token_ids
        - Tensor directly (embeddings)
        - Other formats

        :param result: Result from performant_runner.run()
        :param requests: Original requests to match results
        :return: List of EmbeddingResponse objects
        """
        num_requests = len(requests)

        # If result is a ttnn.Tensor, convert to torch.Tensor first
        # Check for TTNN tensor by type name (could be ttnn.Tensor or ttnn._ttnn.tensor.Tensor)
        result_type_str = str(type(result))
        is_ttnn_tensor = (
            "ttnn" in result_type_str
            and "Tensor" in result_type_str
            and not isinstance(result, torch.Tensor)
        )
        if is_ttnn_tensor:
            try:
                result = ttnn.to_torch(result)
            except Exception as e:
                raise ValueError(f"Unable to convert ttnn.Tensor result: {e}") from e

        # If result is a tensor, extract embeddings directly
        if isinstance(result, torch.Tensor):
            # Extract only the actual request results (excluding padding)
            embeddings = result[:num_requests]
            # Convert to list of Python lists (JSON-serializable)
            if embeddings.dim() == 2:
                # Shape: [batch_size, embedding_dim]
                embedding_list = [emb.cpu().numpy().tolist() for emb in embeddings]
            else:
                # Handle other tensor shapes
                embedding_list = [emb.cpu().numpy().tolist() for emb in embeddings]

            # Use 0 as default token count since we don't have tokenization
            token_counts = [0] * len(requests)

            return [
                EmbeddingResponse(
                    embedding=emb,
                    total_tokens=token_count,
                )
                for emb, token_count in zip(embedding_list, token_counts)
            ]

        # If result is a list, try to extract from objects
        if isinstance(result, list):
            responses = []
            for i, output in enumerate(result[:num_requests]):
                try:
                    # Try to access .outputs.embedding structure
                    if hasattr(output, "outputs") and hasattr(
                        output.outputs, "embedding"
                    ):
                        embedding = output.outputs.embedding
                        embedding = self._ensure_list_format(embedding)
                        if hasattr(output, "prompt_token_ids"):
                            total_tokens = len(output.prompt_token_ids)
                        else:
                            total_tokens = 0
                        responses.append(
                            EmbeddingResponse(
                                embedding=embedding, total_tokens=total_tokens
                            )
                        )
                    elif hasattr(output, "embedding"):
                        # Direct embedding attribute
                        embedding = output.embedding
                        embedding = self._ensure_list_format(embedding)
                        total_tokens = 0
                        responses.append(
                            EmbeddingResponse(
                                embedding=embedding, total_tokens=total_tokens
                            )
                        )
                    else:
                        # If output is a tensor or array, use it directly
                        # Handle TTNN tensors
                        output_type_str = str(type(output))
                        is_ttnn_tensor = (
                            "ttnn" in output_type_str
                            and "Tensor" in output_type_str
                            and not isinstance(output, torch.Tensor)
                        )
                        if is_ttnn_tensor:
                            output = ttnn.to_torch(output)
                        if isinstance(output, torch.Tensor):
                            embedding = output.cpu().numpy().tolist()
                        else:
                            embedding = self._ensure_list_format(output)
                        total_tokens = 0
                        responses.append(
                            EmbeddingResponse(
                                embedding=embedding, total_tokens=total_tokens
                            )
                        )
                except Exception as e:
                    raise ValueError(
                        f"Unable to extract embedding from result item {i}: {e}"
                    ) from e

            return responses

        # If result is a single object, try to extract from it
        try:
            if hasattr(result, "outputs") and hasattr(result.outputs, "embedding"):
                embedding = result.outputs.embedding
                embedding = self._ensure_list_format(embedding)
                if hasattr(result, "prompt_token_ids"):
                    total_tokens = len(result.prompt_token_ids)
                else:
                    total_tokens = 0
                return [
                    EmbeddingResponse(embedding=embedding, total_tokens=total_tokens)
                ]
        except Exception:
            pass

        # Final fallback: try to convert if it looks like a tensor (has shape attribute)
        if hasattr(result, "shape") and not isinstance(result, torch.Tensor):
            try:
                result = ttnn.to_torch(result)
                # Recursively process the converted result
                return self._process_results(result, requests)
            except Exception:
                pass

        # If we get here, we couldn't process the result
        raise ValueError(
            f"Unable to process result of type {type(result)}. "
            f"Expected tensor, list, or object with .outputs.embedding attribute."
        )
