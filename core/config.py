from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    bmtc_api_url: str = "https://nammabmtc.mobi/api/buslocations"
    poll_interval_seconds: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
