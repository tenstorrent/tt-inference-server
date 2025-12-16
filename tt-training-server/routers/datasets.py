# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import yaml
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from config.constants import AVAILABLE_DATASETS

router = APIRouter()

DATASET_BASE_PATH = Path("/datasets")

@router.get("/", response_model=Dict[str, List[str]])
async def list_available_datasets():
    """
    Returns the list of available datasets defined in config/constants.py.
    """
    return {"datasets": AVAILABLE_DATASETS}


