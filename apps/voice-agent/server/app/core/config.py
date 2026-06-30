from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://voicelab:voicelab@localhost:5433/voicelab"
    adk_database_url: str = "postgresql+asyncpg://voicelab:voicelab@localhost:5433/voicelab"
    gemini_api_key: str | None = None
    stt_provider: str = "deepgram"
    stt_api_key: str | None = None
    tts_provider: str = "deepgram"
    tts_api_key: str | None = None
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://localhost:5174", alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    @property
    def effective_tts_api_key(self) -> str | None:
        return self.tts_api_key or self.stt_api_key

    def validate_runtime_settings(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        if self.stt_provider != "deepgram":
            raise ValueError(f"Unsupported STT provider: {self.stt_provider}")
        if self.tts_provider != "deepgram":
            raise ValueError(f"Unsupported TTS provider: {self.tts_provider}")
        if not self.stt_api_key:
            raise ValueError("STT_API_KEY environment variable not set")
        if not self.effective_tts_api_key:
            raise ValueError("TTS_API_KEY or STT_API_KEY environment variable not set")


@lru_cache
def get_settings() -> Settings:
    return Settings()
