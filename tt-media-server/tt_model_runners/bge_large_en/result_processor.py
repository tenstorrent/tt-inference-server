# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import torch
import ttnn
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest

from tt_model_runners.bge_large_en.embedding_converter import convert_to_list


class BGEResultProcessor:
    """Processes results from BGEPerformantRunner into EmbeddingResponse objects."""

    def process(
        self,
        result,
        requests: list[TextEmbeddingRequest],
        token_counts: list[int] = None,
    ) -> list[EmbeddingResponse]:
        """
        Process the result from performant_runner.run() and convert to EmbeddingResponse.

        :param result: Result from performant_runner.run()
        :param requests: Original requests to match results
        :param token_counts: List of token counts for each request (excluding padding)
        :return: List of EmbeddingResponse objects
        """
        num_requests = len(requests)
        token_counts = self._normalize_token_counts(token_counts, num_requests)
        result = self._normalize_result(result)

        if isinstance(result, torch.Tensor):
            return self._process_tensor_result(result, num_requests, token_counts)

        if isinstance(result, list):
            return self._process_list_result(result, num_requests, token_counts)

        if self._is_single_object_result(result):
            return self._process_single_object_result(result, token_counts)

        raise ValueError(
            f"Unable to process result of type {type(result)}. "
            f"Expected tensor, list, or object with .outputs.embedding attribute."
        )

    def _normalize_token_counts(
        self, token_counts: list[int] | None, num_requests: int
    ) -> list[int]:
        """Normalize token counts to match number of requests."""
        if token_counts is None:
            return [0] * num_requests
        return token_counts[:num_requests]

    def _normalize_result(self, result):
        """Convert TTNN tensors to PyTorch tensors."""
        if self._is_ttnn_tensor(result):
            try:
                return ttnn.to_torch(result)
            except Exception as e:
                raise ValueError(f"Unable to convert ttnn.Tensor result: {e}") from e
        return result

    def _process_tensor_result(
        self,
        result: torch.Tensor,
        num_requests: int,
        token_counts: list[int],
    ) -> list[EmbeddingResponse]:
        """Process tensor result."""
        embeddings = result[:num_requests]
        embedding_list = [emb.cpu().numpy().tolist() for emb in embeddings]

        return [
            EmbeddingResponse(embedding=emb, total_tokens=token_count)
            for emb, token_count in zip(embedding_list, token_counts)
        ]

    def _process_list_result(
        self,
        result: list,
        num_requests: int,
        token_counts: list[int],
    ) -> list[EmbeddingResponse]:
        """Process list result."""
        responses = []
        for i, output in enumerate(result[:num_requests]):
            try:
                total_tokens = token_counts[i] if i < len(token_counts) else 0
                embedding = self._extract_embedding_from_output(output, total_tokens)
                responses.append(
                    EmbeddingResponse(embedding=embedding["embedding"], total_tokens=embedding["total_tokens"])
                )
            except Exception as e:
                raise ValueError(
                    f"Unable to extract embedding from result item {i}: {e}"
                ) from e
        return responses

    def _extract_embedding_from_output(self, output, default_token_count: int) -> dict:
        """Extract embedding and token count from output object."""
        total_tokens = default_token_count

        if hasattr(output, "outputs") and hasattr(output.outputs, "embedding"):
            embedding = convert_to_list(output.outputs.embedding)
            if hasattr(output, "prompt_token_ids"):
                total_tokens = len(output.prompt_token_ids)
            return {"embedding": embedding, "total_tokens": total_tokens}

        if hasattr(output, "embedding"):
            embedding = convert_to_list(output.embedding)
            return {"embedding": embedding, "total_tokens": total_tokens}

        # Handle tensor/array output
        if self._is_ttnn_tensor(output):
            output = ttnn.to_torch(output)
        if isinstance(output, torch.Tensor):
            embedding = output.cpu().numpy().tolist()
        else:
            embedding = convert_to_list(output)

        return {"embedding": embedding, "total_tokens": total_tokens}

    def _process_single_object_result(
        self, result, token_counts: list[int]
    ) -> list[EmbeddingResponse]:
        """Process single object result."""
        if hasattr(result, "outputs") and hasattr(result.outputs, "embedding"):
            embedding = convert_to_list(result.outputs.embedding)
            total_tokens = token_counts[0] if token_counts else 0
            if hasattr(result, "prompt_token_ids"):
                total_tokens = len(result.prompt_token_ids)
            return [EmbeddingResponse(embedding=embedding, total_tokens=total_tokens)]
        return []

    def _is_single_object_result(self, result) -> bool:
        """Check if result is a single object with embedding."""
        try:
            return hasattr(result, "outputs") and hasattr(result.outputs, "embedding")
        except Exception:
            return False

    def _is_ttnn_tensor(self, obj) -> bool:
        """Check if object is a TTNN tensor."""
        type_str = str(type(obj))
        return (
            "ttnn" in type_str
            and "Tensor" in type_str
            and not isinstance(obj, torch.Tensor)
        )
