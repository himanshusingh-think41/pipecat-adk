import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from app.runtime.bot import generate_agent_response, run_bot
from app.schemas.session import ChatTurnRequest, ChatTurnResponse, SessionStartResponse

router = APIRouter()
active_sessions: dict[str, dict] = {}
small_webrtc_handler = SmallWebRTCRequestHandler(ice_servers=None)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "voice-agent"}


@router.post("/api/chat", response_model=ChatTurnResponse)
async def chat(payload: ChatTurnRequest) -> ChatTurnResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        assistant_text = await generate_agent_response(
            session_id=payload.session_id,
            user_text=message,
            user_id=payload.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected runtime failure: {exc}",
        ) from exc

    return ChatTurnResponse(
        session_id=payload.session_id,
        user_text=message,
        assistant_text=assistant_text,
    )


@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: Request) -> SessionStartResponse:
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request_data

    ice_config = None
    if request_data.get("enableDefaultIceServers"):
        ice_config = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    return SessionStartResponse(sessionId=session_id, iceConfig=ice_config)


@router.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def on_connection(connection: SmallWebRTCConnection):
        session_id = str(uuid.uuid4())
        background_tasks.add_task(run_bot, connection, session_id=session_id)

    return await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


@router.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@router.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_session(
    session_id: str,
    path: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
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

                async def on_connection(connection: SmallWebRTCConnection):
                    background_tasks.add_task(run_bot, connection, session_id=session_id)

                return await small_webrtc_handler.handle_web_request(
                    request=webrtc_request,
                    webrtc_connection_callback=on_connection,
                )
            if request.method == "PATCH":
                patch_request = SmallWebRTCPatchRequest(
                    pc_id=body["pc_id"],
                    candidates=[IceCandidate(**candidate) for candidate in body.get("candidates", [])],
                )
                await small_webrtc_handler.handle_patch_request(patch_request)
                return {"status": "success"}
        except Exception:
            return Response(content="Invalid WebRTC request", status_code=400)

    return Response(status_code=200)
