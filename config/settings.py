from pydantic_settings import BaseSettings
from pydantic import Field


class AppConfig(BaseSettings):
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_sec: int = 20
    llm_max_retries: int = 3

    max_file_mb: int = 200
    max_rows_preview: int = 100_000
    max_rows_llm_context: int = 50
    max_cols_llm_context: int = 20

    app_name: str = "DataForge AI"
    app_version: str = "1.0.0"
    app_icon: str = "🔬"

    model_config = {"extra": "ignore", "populate_by_name": True}


config = AppConfig()
