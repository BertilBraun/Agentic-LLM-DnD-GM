from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = ''
    redis_url: str = 'redis://redis:6379/0'
    jwt_secret: str = 'changeme'
    jwt_access_expire_minutes: int = 15
    jwt_refresh_expire_days: int = 30
    state_mcp_url: str = 'http://state-mcp:8001'
    media_mcp_url: str = 'http://media-mcp:8004'
    media_root: str = '/media'
    character_creator_url: str = 'http://character-creator:8010'
    campaign_designer_url: str = 'http://campaign-designer:8011'
    dm_agent_url: str = 'http://dm-agent:8012'
    npc_agent_url: str = 'http://npc-agent:8013'
    memory_agent_url: str = 'http://memory-agent:8014'

    model_config = {'env_file': '.env'}


settings = Settings()
