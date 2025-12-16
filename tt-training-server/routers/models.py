# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.constants import AVAILABLE_MODELS

# ------------------------------------------------------------------
# Config: Hugging Face Base Models & Local Storage
# ------------------------------------------------------------------

# Where we save the OUTPUT of fine-tuning (adapters)
FT_MODELS_DIR = Path("/storage/fine_tuned")


router = APIRouter()

# ------------------------------------------------------------------
# Response Schemas
# ------------------------------------------------------------------

class FineTunedModelItem(BaseModel):
    id: str
    name: str
    base_model_id: str
    created_at: Optional[datetime] = None

class FineTunedModelsResponse(BaseModel):
    base_model_id: str
    data: List[FineTunedModelItem]

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

# def find_adapters_for_base(target_base_id: str) -> List[Dict]:
#     """
#     Scans /storage/fine_tuned and returns only adapters 
#     that claim 'target_base_id' as their parent in metadata.json.
#     """
#     if not FT_MODELS_DIR.exists():
#         return []

#     matching_adapters = []
    
#     # Iterate through all folders in the storage directory
#     for model_dir in FT_MODELS_DIR.iterdir():
#         if model_dir.is_dir():
#             meta_path = model_dir / "metadata.json"
            
#             # We strictly need metadata to know the parent
#             if meta_path.exists():
#                 try:
#                     with open(meta_path, "r") as f:
#                         meta = json.load(f)
                        
#                     # THE FILTERING LOGIC
#                     # Only include if the parent matches the requested ID
#                     if meta.get("base_model_id") == target_base_id:
                        
#                         # Try to get creation time
#                         created_at = None
#                         try:
#                             created_at = datetime.fromtimestamp(model_dir.stat().st_ctime)
#                         except:
#                             pass

#                         matching_adapters.append({
#                             "id": model_dir.name,
#                             "name": meta.get("name", model_dir.name),
#                             "base_model_id": target_base_id,
#                             "created_at": created_at
#                         })
#                 except Exception:
#                     # If JSON is corrupt, we skip this folder
#                     continue

#     return matching_adapters

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/base", response_model=Dict[str, List[str]])
async def list_base_models():
    """
    Returns the static list of supported Hugging Face base models.
    """
    return {"data": AVAILABLE_MODELS}


@router.get("/{model_id}/fine_tuned", response_model=FineTunedModelsResponse)
async def list_fine_tuned_models(model_id: str):
    """
    Returns a list of local fine-tuned adapters that belong to the specific base model.
    """
    is_valid_base = any(m["id"] == model_id for m in AVAILABLE_MODELS)
    if not is_valid_base:
        raise HTTPException(
            status_code=404, 
            detail=f"Base model '{model_id}' is not supported."
        )

    adapters = find_adapters_for_base(model_id)

    return {
        "base_model_id": model_id,
        "data": adapters
    }

# TODO: add for a specific fine-tuned model to list the available checkpoints with corresponding metrics