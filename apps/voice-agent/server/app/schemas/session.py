from pydantic import BaseModel


class SessionStartResponse(BaseModel):
    sessionId: str
    iceConfig: dict | None = None


class ChatTurnRequest(BaseModel):
    session_id: str
    message: str
    user_id: str = "local-user"


class ChatTurnResponse(BaseModel):
    session_id: str
    user_text: str
    assistant_text: str
