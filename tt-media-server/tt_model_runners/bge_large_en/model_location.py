# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from typing import Union


def model_location_generator(
    model_version: str,
    model_subdir: str = "",
    download_if_ci_v2: bool = False,
    ci_v2_timeout_in_s: int = 300,
    endpoint_prefix: str = "http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
    download_dir_suffix: str = "model_weights",
) -> Union[Path, str]:
    """
    Determine the appropriate file path for a model based on available locations.

    Checks locations in order:
    1. Internal MLPerf path (/mnt/MLPerf/tt_dnn-models/...)
    2. Falls back to model_version string (uses HuggingFace cache)

    :param model_version: The version identifier of the model (e.g., "BAAI/bge-large-en-v1.5")
    :param model_subdir: Subdirectory within the model folder structure
    :param download_if_ci_v2: Not used (kept for API compatibility)
    :param ci_v2_timeout_in_s: Not used (kept for API compatibility)
    :param endpoint_prefix: Not used (kept for API compatibility)
    :param download_dir_suffix: Not used (kept for API compatibility)
    :return: Path to model files (MLPerf path or model_version string for HF cache)
    """
    model_folder = Path("tt_dnn-models") / model_subdir
    internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version

    if internal_weka_path.exists():
        return internal_weka_path

    # Return model_version string, which will use HuggingFace cache
    # (HF_HOME or ~/.cache/huggingface by default)
    return model_version
