# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
from typing import List

import numpy as np
import torch

from domain.lidar_detection_request import LidarDetectionRequest
from domain.lidar_detection_response import (
    BoundingBox3D,
    LidarDetectionResponse,
    TASK_NAMES,
    TaskDetections,
)
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.forge_runners.runners import ensure_model_loaders
from utils.decorators import log_execution_time

os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

xla_backend = "tt"


def _import_centerpoint_model():
    """Import CenterPoint model components after ensuring model loaders exist."""
    ensure_model_loaders()
    from tt_model_runners.forge_runners.model_loaders.centerpoint.pytorch.src.model import (
        GRID_X,
        GRID_Y,
        PillarFeatureNetCPU,
        _PointPillarsScatterCPU,
        _load_checkpoint,
        _remap_pfn_keys,
        get_single_input,
        load_model_with_weights,
        postprocess,
        voxelize,
    )

    return {
        "load_model_with_weights": load_model_with_weights,
        "PillarFeatureNetCPU": PillarFeatureNetCPU,
        "ScatterCPU": _PointPillarsScatterCPU,
        "load_checkpoint": _load_checkpoint,
        "remap_pfn_keys": _remap_pfn_keys,
        "get_single_input": get_single_input,
        "postprocess": postprocess,
        "voxelize": voxelize,
        "GRID_X": GRID_X,
        "GRID_Y": GRID_Y,
    }


def _build_response(detections: List[dict]) -> LidarDetectionResponse:
    task_results = []
    total = 0
    for task_idx, det in enumerate(detections):
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]
        task_name = TASK_NAMES[task_idx] if task_idx < len(TASK_NAMES) else f"task_{task_idx}"
        box_list = []
        for j in range(len(boxes)):
            b = boxes[j]
            box_list.append(
                BoundingBox3D(
                    cx=b[0].item(),
                    cy=b[1].item(),
                    cz=b[2].item(),
                    width=b[3].item(),
                    length=b[4].item(),
                    height=b[5].item(),
                    yaw=b[6].item(),
                    vx=b[7].item(),
                    vy=b[8].item(),
                    score=scores[j].item(),
                    label=labels[j].item(),
                )
            )
        task_results.append(TaskDetections(task_name=task_name, detections=box_list))
        total += len(box_list)

    return LidarDetectionResponse(tasks=task_results, num_detections=total)


class ForgeCenterpointRunner(BaseDeviceRunner):
    """CenterPoint 3D object detection runner.

    CPU pipeline:
        raw .bin bytes → voxelize → PillarFeatureNet → PointPillarsScatter → BEV

    TT pipeline:
        BEV (1, 64, 512, 512) → RPN + CenterHead → task_outputs

    CPU postprocess:
        task_outputs → 3D bounding boxes
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.dtype = torch.bfloat16
        self.pfn = None
        self.scatter = None
        self.compiled_model = None
        self.device = None
        self._grid_x = None
        self._grid_y = None
        self._voxelize = None
        self._postprocess = None

    def load_weights(self) -> bool:
        # CenterPoint downloads weights from OpenMMLab, not HuggingFace.
        # Return True so the HuggingFace utility skips its download flow.
        return True

    @log_execution_time("CenterPoint model warmup")
    async def warmup(self) -> bool:
        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        use_optimizer = os.getenv("USE_OPTIMIZER", "true").lower() == "true"

        cp = _import_centerpoint_model()

        self._voxelize = cp["voxelize"]
        self._postprocess = cp["postprocess"]
        self._grid_x = cp["GRID_X"]
        self._grid_y = cp["GRID_Y"]

        self.logger.info("Loading PillarFeatureNet weights on CPU ...")
        self.pfn = cp["PillarFeatureNetCPU"]()
        sd = cp["load_checkpoint"]()
        self.pfn.load_state_dict(cp["remap_pfn_keys"](sd), strict=False)
        self.pfn.eval()

        self.scatter = cp["ScatterCPU"]()

        self.logger.info(f"Loading CenterPointRPNHead on device {self.device_id} ...")
        model = cp["load_model_with_weights"](dtype=self.dtype)

        if runs_on_cpu:
            self.device = torch.device("cpu")
            self.compiled_model = model.to(self.device)
        else:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr

            xr.set_device_type("TT")
            self.device = xm.xla_device()
            self.logger.info("## Compiling CenterPoint RPN+Head ##")
            torch_xla.set_custom_compile_options(
                {
                    "enable_optimizer": use_optimizer,
                    "enable_fusing_conv2d_with_multiply_pattern": use_optimizer,
                }
            )
            model.compile(backend=xla_backend)
            self.compiled_model = model.to(self.device)

        self.logger.info("## CenterPoint warmup inference ##")
        bev = cp["get_single_input"](dtype=self.dtype, batch_size=1).to(self.device)
        with torch.no_grad():
            _ = self.compiled_model(bev)

        return True

    @log_execution_time("CenterPoint inference")
    def run(self, requests: List[LidarDetectionRequest]):
        self.logger.info(f"CenterPoint inference on device {self.device_id}")

        if not requests:
            raise ValueError("Empty requests list provided")

        if len(requests) > 1:
            self.logger.warning(
                f"Batch processing not supported; processing only first of {len(requests)} requests"
            )

        request = requests[0]
        points = np.frombuffer(request.prompt, dtype=np.float32).reshape(-1, 5)

        voxels_np, coords_np, num_pts_np = self._voxelize(points)
        voxels = torch.from_numpy(voxels_np)
        coords = torch.from_numpy(coords_np)
        num_pts = torch.from_numpy(num_pts_np)

        with torch.no_grad():
            pillar_features = self.pfn(voxels, num_pts, coords)
            bev = self.scatter(pillar_features, coords, 1, self._grid_y, self._grid_x)

        bev = bev.to(self.dtype).to(self.device)

        with torch.no_grad():
            task_outputs = self.compiled_model(bev)

        task_outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in task_outputs]
        detections = self._postprocess(
            task_outputs_cpu,
            score_threshold=request.score_threshold,
        )

        return [_build_response(detections)]
