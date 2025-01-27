# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import logging
from fastapi import FastAPI, File, HTTPException, Request, status, UploadFile
from fastapi.responses import JSONResponse
from functools import wraps
from io import BytesIO
import jwt
from PIL import Image
from models.demos.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ
import ttnn
from typing import Optional

import numpy as np
import torch
import time

app = FastAPI(
    title="YOLOv4 object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_dispatch_core_type():
    # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    return dispatch_core_type


@app.on_event("startup")
async def startup():
    global class_names

    def load_class_names(namesfile):
        class_names = []
        with open(namesfile, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                class_names.append(line)
        return class_names

    namesfile = "coco.names"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, namesfile)
    class_names = load_class_names(file_path)

    global model
    global ready
    ready = False
    if ("WH_ARCH_YAML" in os.environ) and os.environ[
        "WH_ARCH_YAML"
    ] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        print("WH_ARCH_YAML:", os.environ.get("WH_ARCH_YAML"))
        device_id = 0
        dispatch_core_config = ttnn.DispatchCoreConfig(
            get_dispatch_core_type(), ttnn.DispatchCoreAxis.ROW
        )
        device = ttnn.CreateDevice(
            device_id,
            dispatch_core_config=dispatch_core_config,
            l1_small_size=24576,
            trace_region_size=3211264,
            num_command_queues=2,
        )
        ttnn.enable_program_cache(device)
        model = Yolov4Trace2CQ()
        model.initialize_yolov4_trace_2cqs_inference(device)
    else:
        device_id = 0
        device = ttnn.CreateDevice(
            device_id,
            l1_small_size=24576,
            trace_region_size=3211264,
            num_command_queues=2,
        )
        ttnn.enable_program_cache(device)
        model = Yolov4Trace2CQ()
        model.initialize_yolov4_trace_2cqs_inference(device)
    ready = True


@app.get("/health")
async def health_check():
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is not ready yet",
        )
    return JSONResponse(content={"message": "OK\n"}, status_code=status.HTTP_200_OK)


@app.on_event("shutdown")
async def shutdown():
    model.release_yolov4_trace_2cqs_inference()


def process_output(output):
    outs = []
    output = output
    cnt = 0
    for item in output:
        cnt = cnt + 1
        output_i = [element.item() for element in item]
        output_i.append(class_names[output_i[-1]])  # map class id to class name
        outs.append(output_i)
    return outs


def post_processing(img, conf_thresh, nms_thresh, output):
    box_array = output[0]
    confs = output[1]

    box_array = np.array(box_array.to(torch.float32))
    confs = np.array(confs.to(torch.float32))

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[k, 0],
                            ll_box_array[k, 1],
                            ll_box_array[k, 2],
                            ll_box_array[k, 3],
                            ll_max_conf[k],
                            ll_max_id[k],
                        ]
                    )

        bboxes_batch.append(bboxes)

    return bboxes_batch


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def normalize_token(token) -> [str, str]:
    """
    Note that scheme is case insensitive for the authorization header.
    See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization#directives
    """  # noqa: E501
    one_space = " "
    words = token.split(one_space)
    scheme = words[0].lower()
    return [scheme, " ".join(words[1:])]


def read_authorization(
    headers,
) -> Optional[dict]:
    authorization = headers.get("authorization")
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Must provide Authorization header.",
        )
    [scheme, parameters] = normalize_token(authorization)
    if scheme != "bearer":
        user_error_msg = f"Authorization scheme was '{scheme}' instead of bearer"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=user_error_msg
        )
    try:
        payload = jwt.decode(parameters, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        if not payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return payload
    except jwt.InvalidTokenError as exc:
        user_error_msg = f"JWT payload decode error: {exc}"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=user_error_msg
        )


def api_key_required(f):
    """Decorates an endpoint to require API key validation"""

    @wraps(f)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        _ = read_authorization(request.headers)

        return await f(*args, **kwargs)

    return wrapper


@app.post("/objdetection_v2")
@api_key_required
async def objdetection_v2(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = image.resize((320, 320))  # Resize to target dimensions
    image = np.array(image)
    if isinstance(image, np.ndarray) and len(image.shape) == 3:  # cv2 image
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    elif isinstance(image, np.ndarray) and len(image.shape) == 4:
        image = torch.from_numpy(image).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()
    response = model.run_traced_inference(image)
    t2 = time.time()
    logging.info("The inference on the sever side took: %.3f seconds", t2 - t1)
    conf_thresh = 0.6
    nms_thresh = 0.5

    boxes = post_processing(image, conf_thresh, nms_thresh, response)
    output = boxes[0]
    # output = boxes
    try:
        output = process_output(output)
    except Exception as E:
        print("the Exception is: ", E)
        print("No objects detected!")
        return []
    t3 = time.time()
    logging.info("The post-processing to get the boxes took: %.3f seconds", t3 - t2)

    return output
