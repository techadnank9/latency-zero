import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    local_model: str = os.getenv("LOCAL_MODEL", "google/functiongemma-270m-it")
    cloud_model: str = os.getenv("CLOUD_MODEL", "gemini-2.0-flash")
    transcribe_model: str = os.getenv("TRANSCRIBE_MODEL", "openai/whisper-small")
    cactus_bin: str = os.getenv("CACTUS_BIN", "")
    cactus_cwd: str = os.getenv("CACTUS_CWD", "")


def get_settings() -> Settings:
    return Settings()
