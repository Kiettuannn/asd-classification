"""Tests for application settings and configuration."""

from src.config.settings import get_settings


def test_settings_initialization():
    """Verify settings defaults and directory properties."""
    settings = get_settings()
    assert settings.atlas_name == "cc200"
    assert settings.num_rois == 200
    assert settings.fc_threshold == 0.3
    assert settings.DATA_DIR.exists()
    assert settings.ARTIFACTS_DIR.exists()
