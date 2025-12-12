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


@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset_configuration(dataset_id: str):
    """
    Returns the parsed YAML configuration for the requested dataset.
    """
    if dataset_id not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404, 
            detail=f"Dataset '{dataset_id}' is not recognized in the registry."
        )

    config_path = DATASET_BASE_PATH / dataset_id / "config.yaml"

    if not config_path.exists():
        raise HTTPException(
            status_code=500, 
            detail=f"Configuration file missing on disk for '{dataset_id}' at {config_path}"
        )

    try:
        with open(config_path, "r") as f:
            config_content = yaml.safe_load(f)
            return config_content
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse YAML configuration for '{dataset_id}': {exc}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=str(e)
        )