"""
Configuration management for the Multi-Business Conversational Chatbot.
Handles environment variables, validation, and provides typed configuration.
"""

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from dotenv import load_dotenv
import asyncio
from bson import ObjectId
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.services.mongodb_service import MongoDBService
from backend.app.config import Settings
from backend.app.models.business import BusinessSchema

# Load environment variables
load_dotenv()

class BusinessConfig(BaseSettings):
    """Individual business configuration"""
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    
    @field_validator('port', mode='before')
    def validate_port(cls, v):
        return int(v) if isinstance(v, str) else v

class MongoDBConfig(BaseSettings):
    """MongoDB configuration settings"""
    uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    database: str = Field(default="chatbot_config", env="MONGODB_DB")
    user: Optional[str] = Field(default=None, env="MONGODB_USER")
    password: Optional[str] = Field(default=None, env="MONGODB_PASSWORD")
    
    class Config:
        env_prefix = "MONGODB_"

class RedisConfig(BaseSettings):
    """Redis configuration settings"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    class Config:
        env_prefix = "REDIS_"

class JWTSettings(BaseSettings):
    """JWT configuration settings"""
    secret_key: str = Field(env="JWT_SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    expiration: int = Field(default=3600, env="JWT_EXPIRATION")
    refresh_expiration: int = Field(default=86400, env="JWT_REFRESH_EXPIRATION")
    
    class Config:
        env_prefix = "JWT_"

class MistralConfig(BaseSettings):
    """Mistral LLM configuration settings"""
    api_key: str = Field(env="MISTRAL_API_KEY")
    model: str = Field(default="mistral-large-2407", env="MISTRAL_MODEL")
    class Config:
        env_prefix = "MISTRAL_"

class VectorSearchConfig(BaseSettings):
    """Vector search configuration settings"""
    store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    faiss_index_path: Optional[str] = Field(default=None, env="FAISS_INDEX_PATH")
    class Config:
        env_prefix = ""

class ServerConfig(BaseSettings):
    """Server configuration settings"""
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CORS settings
    allowed_origins: List[str] = Field(default=["http://localhost:3000"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")
    
    @field_validator('allowed_origins', 'allowed_methods', 'allowed_headers', mode='before')
    def parse_list_fields(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v

class SecurityConfig(BaseSettings):
    """Security configuration settings"""
    bcrypt_rounds: int = Field(default=12, env="BCRYPT_ROUNDS")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    class Config:
        env_prefix = ""

class FeatureFlags(BaseSettings):
    """Feature flags configuration"""
    enable_vector_search: bool = Field(default=True, env="ENABLE_VECTOR_SEARCH")
    enable_schema_discovery: bool = Field(default=True, env="ENABLE_SCHEMA_DISCOVERY")
    enable_conversation_memory: bool = Field(default=True, env="ENABLE_CONVERSATION_MEMORY")
    enable_multi_business: bool = Field(default=True, env="ENABLE_MULTI_BUSINESS")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    
    class Config:
        env_prefix = ""

class Settings(BaseSettings):
    """Main application settings"""
    
    # Core configurations
    mongodb: MongoDBConfig = MongoDBConfig()
    redis: RedisConfig = RedisConfig()
    jwt: JWTSettings = JWTSettings()
    vector_search: VectorSearchConfig = VectorSearchConfig()
    server: ServerConfig = ServerConfig()
    security: SecurityConfig = SecurityConfig()
    features: FeatureFlags = FeatureFlags()
    mistral: MistralConfig = MistralConfig()
    
    # Multi-business configuration
    business_ids: List[str] = Field(default=[], env="BUSINESS_IDS")
    business_configs: Dict[str, BusinessConfig] = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_business_configs()
    
    def _load_business_configs(self):
        """Load business-specific configurations from environment variables"""
        for business_id in self.business_ids:
            if not business_id.strip():
                continue
                
            # Load business-specific config
            prefix = f"BUSINESS_{business_id.upper()}_"
            config = BusinessConfig(
                host=os.getenv(f"{prefix}POSTGRES_HOST"),
                database=os.getenv(f"{prefix}POSTGRES_DB"),
                user=os.getenv(f"{prefix}POSTGRES_USER"),
                password=os.getenv(f"{prefix}POSTGRES_PASSWORD"),
                port=int(os.getenv(f"{prefix}POSTGRES_PORT", "5432"))
            )
            
            # Validate that all required config is present
            if all([config.host, config.database, config.user, config.password]):
                self.business_configs[business_id] = config
            else:
                raise ValueError(f"Incomplete configuration for business: {business_id}")
    
    @field_validator('business_ids', mode='before')
    def parse_business_ids(cls, v):
        if isinstance(v, str):
            return [bid.strip() for bid in v.split(',') if bid.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def get_business_config(business_id: str) -> BusinessConfig:
    """Get configuration for a specific business"""
    if business_id not in settings.business_configs:
        raise ValueError(f"Business '{business_id}' not configured")
    return settings.business_configs[business_id]

def get_database_url(business_id: str) -> str:
    """Get database URL for a specific business"""
    config = get_business_config(business_id)
    return f"postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"

def get_mongodb_url() -> str:
    """Get MongoDB connection URL"""
    config = settings.mongodb
    if config.user and config.password:
        return f"mongodb://{config.user}:{config.password}@{config.uri.replace('mongodb://', '')}/{config.database}"
    return f"{config.uri}/{config.database}"

def get_redis_url() -> str:
    """Get Redis connection URL"""
    config = settings.redis
    if config.password:
        return f"redis://:{config.password}@{config.host}:{config.port}/{config.db}"
    return f"redis://{config.host}:{config.port}/{config.db}"

def validate_business_access(user_id: str, requested_business_id: str) -> bool:
    """Validate if user has access to the requested business by checking MongoDB user permissions."""
    user = asyncio.run(mongodb_service.get_user_by_id(user_id))
    if not user:
        return False
    allowed_businesses = user.get('allowed_businesses', [])
    return requested_business_id in allowed_businesses 

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # This tells Pydantic to treat it as a string in OpenAPI/JSON schema
        return {"type": "string"} 