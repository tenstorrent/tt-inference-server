# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config.constants import AVAILABLE_MODELS
from domain.model_dtos import FineTunedModelsResponse, FineTunedModelItem

# ------------------------------------------------------------------
# Config: Hugging Face Base Models & Local Storage
# ------------------------------------------------------------------

# Where we save the OUTPUT of fine-tuning (adapters)
FT_MODELS_DIR = Path("/storage/fine_tuned")


router = APIRouter()

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/base", response_model=Dict[str, List[str]])
async def list_base_models():
    """
    Returns the static list of supported Hugging Face base models.
    """
    return {"data": AVAILABLE_MODELS}


# @router.get("/{model_id}/fine_tuned", response_model=FineTunedModelsResponse)
# async def list_fine_tuned_models(model_id: str):
#     """
#     Returns a list of local fine-tuned adapters that belong to the specific base model.
#     """
#     is_valid_base = any(m["id"] == model_id for m in AVAILABLE_MODELS)
#     if not is_valid_base:
#         raise HTTPException(
#             status_code=404, 
#             detail=f"Base model '{model_id}' is not supported."
#         )

#     adapters = find_adapters_for_base(model_id)

#     return {
#         "base_model_id": model_id,
#         "data": adapters
#     }

# TODO: add for a specific fine-tuned model to list the available checkpoints with corresponding metrics