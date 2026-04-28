#!/usr/bin/env python3
"""
TT-Home Phone Control PWA — Backend Server
Serves the PWA and proxies requests to TT-Home on port 8080.
"""

import os
import time
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TT_HOME_URL = os.getenv("TT_HOME_URL", "http://localhost:8080")
TT_HOME_PUBLIC_URL = os.getenv("TT_HOME_PUBLIC_URL", "")

app = FastAPI(title="TT-Home Phone Control")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODES = {
    "kids": {
        "id": "kids",
        "name": "Kids Entertainment",
        "subtitle": "Video gen — 4/4 chips",
        "description": "All 4 chips dedicated to video generation for kids content.",
        "chips": 4,
        "available": False,
    },
    "personal": {
        "id": "personal",
        "name": "Personal / Family",
        "subtitle": "Balanced — chat, photos, home help",
        "description": "Voice assistant, document analysis, podcasts, and more.",
        "chips": 4,
        "available": True,
    },
    "professional": {
        "id": "professional",
        "name": "Professional Coding / Agents",
        "subtitle": "LLM/VLM — 4/4 chips",
        "description": "All 4 chips for large language model and vision tasks.",
        "chips": 4,
        "available": False,
    },
    "security": {
        "id": "security",
        "name": "Security / Away",
        "subtitle": "Lockdown — alerts enabled",
        "description": "Home monitoring, sensors, cameras, and alert system.",
        "chips": 4,
        "available": True,
    },
}

current_mode = "personal"
mode_changed_at = time.time()


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(SCRIPT_DIR, "templates", "index.html"))


@app.get("/api/device-status")
async def device_status():
    """Return device info and current mode."""
    tt_home_ok = False
    services = {}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{TT_HOME_URL}/health")
            if r.status_code == 200:
                tt_home_ok = True
                services = r.json()
    except Exception:
        pass

    return {
        "device": "Blackhole Quiet Box 2",
        "model": "TT-QuietBox 2 (TM)",
        "chip_type": "Blackhole (TM)",
        "chips": 4,
        "current_mode": current_mode,
        "mode_info": MODES.get(current_mode),
        "mode_changed_at": mode_changed_at,
        "tt_home_connected": tt_home_ok,
        "services": services,
    }


@app.get("/api/modes")
async def list_modes():
    return {"modes": list(MODES.values()), "current": current_mode}


@app.get("/api/tt-home-url")
async def tt_home_url(request: Request):
    """Return the best URL for the browser to reach TT-Home directly."""
    if TT_HOME_PUBLIC_URL:
        return {"url": TT_HOME_PUBLIC_URL}
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    if "trycloudflare.com" in host or "." in host.split(":")[0]:
        if TT_HOME_PUBLIC_URL:
            return {"url": TT_HOME_PUBLIC_URL}
    origin_host = request.client.host if request.client else "localhost"
    return {"url": f"http://{request.headers.get('host', 'localhost').split(':')[0]}:8080"}


@app.post("/api/set-mode")
async def set_mode(request: Request):
    global current_mode, mode_changed_at
    data = await request.json()
    mode_id = data.get("mode")
    if mode_id not in MODES:
        return JSONResponse({"error": f"Unknown mode: {mode_id}"}, status_code=400)
    current_mode = mode_id
    mode_changed_at = time.time()
    return {"status": "ok", "current_mode": current_mode, "mode_info": MODES[current_mode]}


@app.api_route("/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_tt_home(path: str, request: Request):
    """Forward requests to TT-Home so the phone only needs access to port 8081."""
    target = f"{TT_HOME_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if request.method == "GET":
                r = await client.get(target, headers=headers, params=request.query_params)
            else:
                body = await request.body()
                r = await client.request(
                    request.method, target, headers=headers, content=body,
                    params=request.query_params,
                )
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=dict(r.headers),
            media_type=r.headers.get("content-type"),
        )
    except Exception as e:
        return JSONResponse({"error": f"Proxy error: {e}"}, status_code=502)


app.mount("/static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info", reload=False)
