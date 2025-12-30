# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService


class CNNService(BaseService):
    def __init__(self):
        super().__init__()

    # TODO: implement this
    def run(self, image_search_requests: List[ImageSearchRequest]):
        request = image_search_requests[0]

        # Handle base64 or bytes input
        pil_image = self._get_pil_image(request.prompt)

        inputs = self.loader.input_preprocess(...)
        output = self.compiled_model(inputs)

        # Get predictions with top_k and min_confidence
        # (these are inference-time filters, belong here)
        predictions = self._get_filtered_predictions(
            output, top_k=request.top_k, min_confidence=request.min_confidence
        )

        # Return raw prediction list - let service/API format it
        return [predictions]
