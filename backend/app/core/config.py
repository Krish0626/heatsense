"""Application settings, loaded from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "HeatSense AI"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg://heatsense:heatsense@localhost:5432/heatsense"

    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Directory holding the trained XGBoost model + SHAP explainer artifacts
    # produced by ml/src/models/train_xgboost.py
    ml_model_dir: str = "../ml/models_store"


@lru_cache
def get_settings() -> Settings:
    return Settings()
