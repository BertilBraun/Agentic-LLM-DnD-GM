import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = os.getenv('DATABASE_URL', '')
    port: int = int(os.getenv('PORT', '8001'))

    model_config = {'env_file': '.env'}


settings = Settings()
