# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import io
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from app import config
from app.engine import FaceEngine

_engine: Optional[FaceEngine] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _start_time
    _start_time = time.time()
    _engine = FaceEngine(
        faces_dir=config.faces_dir(),
        device_id=config.device_id(),
        match_threshold=config.recognition_threshold(),
    )
    try:
        _engine.initialize()
    except Exception as e:
        # Still serve /health so ops can see failure reason
        print(f"[face-api] init failed: {e}", flush=True)
    yield
    if _engine is not None:
        _engine.shutdown()


app = FastAPI(title="Face Recognition API", lifespan=lifespan)


def _get_engine() -> FaceEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not allocated")
    return _engine


@app.get("/health")
def health():
    eng = _get_engine()
    return {
        "status": "healthy" if eng.ready else "degraded",
        "models_loaded": eng.ready,
        "registered_faces": len(eng.face_database),
        "device_id": config.device_id(),
        "uptime_seconds": int(time.time() - _start_time),
    }


@app.get("/registered-faces")
def list_faces():
    eng = _get_engine()
    names = eng.list_identities()
    return {"faces": names, "count": len(names)}


@app.delete("/registered-faces/{name}")
def delete_face(name: str):
    eng = _get_engine()
    if not eng.ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    ok = eng.delete_identity(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown identity: {name}")
    return {"success": True, "name": name, "message": "Deleted"}


@app.post("/register-face")
async def register_face(name: str = Form(...), image: UploadFile = File(...)):
    import numpy as np

    eng = _get_engine()
    if not eng.ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="name must be alphanumeric or underscore")
    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw))
        frame_rgb = np.array(pil.convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    ok, msg, conf = eng.register_image_rgb(frame_rgb, name.strip())
    if not ok:
        code = 409 if "already exists" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    return {"success": True, "name": name.strip(), "message": msg, "confidence": conf}


@app.post("/recognize-face")
async def recognize_face(image: UploadFile = File(...)):
    import numpy as np

    eng = _get_engine()
    if not eng.ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw))
        frame_rgb = np.array(pil.convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    faces, ms = eng.recognize_image_rgb(frame_rgb)
    return {"faces": faces, "count": len(faces), "inference_ms": round(ms, 2)}
