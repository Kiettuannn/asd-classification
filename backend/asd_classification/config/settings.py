"""Typed application configuration and settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
except ImportError:
    # Graceful fallback if pydantic-settings is not yet installed in active environment
    from pydantic import BaseModel as BaseSettings, Field  # type: ignore

    class SettingsConfigDict:  # type: ignore
        def __init__(self, **kwargs):
            pass


BACKEND_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Central configuration for datasets, features, models, and API."""

    # Application and shared research directories
    BACKEND_DIR: Path = BACKEND_DIR
    REPOSITORY_ROOT: Path = REPOSITORY_ROOT
    DATA_DIR: Path = REPOSITORY_ROOT / "data"
    ARTIFACTS_DIR: Path = REPOSITORY_ROOT / "artifacts"

    @property
    def RAW_ABIDE_DIR(self) -> Path:
        return self.DATA_DIR / "ABIDE_pcp"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def MODELS_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "models"

    @property
    def REPORTS_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "reports"

    # Dataset & Connectivity Hyperparameters
    atlas_name: str = "cc200"
    num_rois: int = 200
    fc_metric: Literal["correlation", "partial correlation", "tangent"] = "correlation"
    fc_threshold: float = 0.3
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42

    # Classical ML (SVM) Settings
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"

    # Deep Learning & GNN Settings
    device: str = "cuda"
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    gnn_hidden_dim: int = 64
    gat_heads: int = 4
    dropout: float = 0.5

    # API Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "ASD Classification & Explainability API"
    api_version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_prefix="ASD_",
        env_file=".env",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.ARTIFACTS_DIR,
            self.MODELS_DIR,
            self.REPORTS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Returns singleton Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
