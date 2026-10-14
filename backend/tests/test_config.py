"""Tests for application settings and configuration."""

from asd_classification.config.settings import get_settings


def test_settings_initialization():
    """Verify settings defaults and directory properties."""
    settings = get_settings()
    assert settings.atlas_name == "cc200"
    assert settings.num_rois == 200
    assert settings.fc_threshold == 0.3
    assert settings.DATA_DIR == settings.REPOSITORY_ROOT / "data"
    assert settings.ARTIFACTS_DIR == settings.REPOSITORY_ROOT / "artifacts"
    assert settings.DATA_DIR.exists()
    assert settings.ARTIFACTS_DIR.exists()


def test_settings_allow_shared_asset_directories_to_be_overridden(monkeypatch, tmp_path):
    """Storage locations can be supplied by the environment."""
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ASD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ASD_ARTIFACTS_DIR", str(artifacts_dir))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.DATA_DIR == data_dir
    assert settings.ARTIFACTS_DIR == artifacts_dir
    get_settings.cache_clear()
