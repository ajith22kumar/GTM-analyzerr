from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://gtm_user:gtm_password@localhost:5432/gtm_analyzer"
    )

    max_pages: int = 500
    max_depth: int = 10
    crawl_timeout: int = 30          # seconds, per-request
    concurrent_requests: int = 5
    retry_attempts: int = 2
    respect_robots_txt: bool = True
    allow_external_domains: bool = False
    user_agent: str = "GTM-Analyzer-Bot/1.0 (+https://example.com/bot)"

    priority_path_hints: tuple[str, ...] = (
        "home", "about", "product", "service", "solution", "feature",
        "pricing", "customer", "case-stud", "case_stud", "industr",
        "blog", "resource", "partner", "contact", "career",
    )

  
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    embedding_provider: str = "local"  
   
    embedding_model: str = "text-embedding-3-small"  # used by "openai" only
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"  # used by "local" only
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64
    llm_model: str = "claude-sonnet-4-6"

    environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
