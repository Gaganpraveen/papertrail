import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".papertrail")
    model: str = "qwen3.5:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_timeout: float = 240.0
    max_pdf_bytes: int = 30 * 1024 * 1024
    max_pages: int = 100
    candidates: int = 8

    @classmethod
    def from_env(cls, data_dir: Path | None = None, model: str | None = None) -> "Settings":
        return cls(
            data_dir=data_dir or Path(os.getenv("PAPERTRAIL_DATA_DIR", ".papertrail")),
            model=model or os.getenv("PAPERTRAIL_MODEL", "qwen3.5:4b"),
            ollama_url=os.getenv("PAPERTRAIL_OLLAMA_URL", "http://127.0.0.1:11434"),
            model_timeout=float(os.getenv("PAPERTRAIL_MODEL_TIMEOUT", "240")),
        )
