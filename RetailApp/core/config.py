from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    ENV_STATE: str = Field(validation_alias="ENV_STATE")
    LOG_LEVEL: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    SECRET_KEY: str = Field(validation_alias="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", validation_alias="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, validation_alias="REFRESH_TOKEN_EXPIRE_DAYS")

    ALLOWED_ORIGINS: list[str] = Field(default=["http://localhost:8000"], validation_alias="ALLOWED_ORIGINS")

    ML_PRICER_MODEL_PATH: str = Field(validation_alias="ML_PRICER_MODEL_PATH")
    ML_PRICER_RAW_DATA_PATH: str = Field(validation_alias="ML_PRICER_RAW_DATA_PATH")
    ML_PRICER_TRAIN_DATA_PATH: str = Field(validation_alias="ML_PRICER_TRAIN_DATA_PATH")
    ML_PRICER_TEST_DATA_PATH: str = Field(validation_alias="ML_PRICER_TEST_DATA_PATH")
    ML_PRICER_MODEL_VERSION: str = Field(validation_alias="ML_PRICER_MODEL_VERSION")
    ML_PRICER_METRICS_PATH: str = Field(validation_alias="ML_PRICER_METRICS_PATH")


class GlobalConfig(BaseConfig):
    DATABASE_URL: Optional[str] = None
    DB_FORCE_ROLL_BACK: bool = False


class DevConfig(GlobalConfig):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEV_", extra="ignore", case_sensitive=False)


class ProdConfig(GlobalConfig):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PROD_", extra="ignore", case_sensitive=False)


class TestConfig(GlobalConfig):
    DATABASE_URL: Optional[str] = "sqlite+aiosqlite:///:memory:"
    DB_FORCE_ROLL_BACK: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TEST_", extra="ignore", case_sensitive=False)


def get_config(env_state: str) -> GlobalConfig:
    if env_state == "DEV":
        return DevConfig()
    elif env_state == "PROD":
        return ProdConfig()
    elif env_state == "TEST":
        return TestConfig()
    else:
        raise ValueError(f"Invalid ENV_STATE: {env_state}")


config = get_config(BaseConfig().ENV_STATE)
