"""FastAPI server for the assistant bot.

This server handles WebRTC signaling and manages bot instances.
"""

import argparse
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import uvicorn
from bot import run_bot
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

# Load environment variables
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pipecat-server")

app = FastAPI()

# In-memory store of active sessions: session_id -> session info
active_sessions: Dict[str, Dict[str, Any]] = {}

# SmallWebRTC request handler
small_webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=None,  # Uses default STUN servers
)

# Mount the frontend at /client
app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/start")
async def rtvi_start(request: Request):
    """Session init endpoint expected by pipecat-ai-small-webrtc-prebuilt v2+."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request_data

    result: dict = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    return result


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC SDP offer."""
    async def on_connection(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection)

    return await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handle trickle ICE candidates."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_session(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """Proxy route: forwards /sessions/{id}/api/offer to the offer handler."""
    if session_id not in active_sessions:
        return Response(content="Invalid session", status_code=404)

    if path.endswith("api/offer"):
        try:
            body = await request.json()
            if request.method == "POST":
                webrtc_request = SmallWebRTCRequest(
                    sdp=body["sdp"],
                    type=body["type"],
                    pc_id=body.get("pc_id"),
                    restart_pc=body.get("restart_pc"),
                )
                return await offer(webrtc_request, background_tasks)
            elif request.method == "PATCH":
                patch_request = SmallWebRTCPatchRequest(
                    pc_id=body["pc_id"],
                    candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
                )
                return await ice_candidate(patch_request)
        except Exception as e:
            logger.error(f"Failed to parse WebRTC request: {e}")
            return Response(content="Invalid WebRTC request", status_code=400)

    return Response(status_code=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat ADK Assistant Bot")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    uvicorn.run(app, host=args.host, port=args.port)
