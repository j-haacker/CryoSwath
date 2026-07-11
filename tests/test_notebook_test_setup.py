from configparser import ConfigParser
from pathlib import Path

import pytest

notebook_setup = pytest.importorskip("tools.prepare_notebook_tests")


def _write_auxiliary_sentinels(project_dir: Path) -> None:
    for relative_path in notebook_setup.AUXILIARY_SENTINELS:
        target = project_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ready")


def test_prepare_report_project_creates_config_and_downloads_auxiliary(
    monkeypatch, tmp_path
):
    calls = []

    def fake_download_auxiliary_data(base_dir=".", *, force=False, timeout=120):
        calls.append((Path(base_dir), force, timeout))
        _write_auxiliary_sentinels(Path(base_dir))
        return str(Path(base_dir) / "data" / "auxiliary")

    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        fake_download_auxiliary_data,
    )

    project = notebook_setup.prepare_report_project(tmp_path / "reports", timeout=7)

    config = ConfigParser()
    config.read(project.config_path)
    assert config["path"]["data"] == "data"
    assert calls == [(tmp_path / "reports", False, 7)]


def test_prepare_report_project_writes_external_dem_path(monkeypatch, tmp_path):
    dem_dir = tmp_path / "external-dem"
    dem_dir.mkdir()

    def fake_download_auxiliary_data(base_dir=".", *, force=False, timeout=120):
        _write_auxiliary_sentinels(Path(base_dir))
        return str(Path(base_dir) / "data" / "auxiliary")

    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        fake_download_auxiliary_data,
    )

    project = notebook_setup.prepare_report_project(
        tmp_path / "reports",
        dem_path=dem_dir,
    )

    config = ConfigParser()
    config.read(project.config_path)
    assert config["path"]["dem"] == str(dem_dir.resolve())


def test_prepare_report_project_clears_stale_dem_path(monkeypatch, tmp_path):
    project_dir = tmp_path / "reports"
    project_dir.mkdir()
    config_path = project_dir / "cryoswath.cfg"
    config_path.write_text("[path]\ndata = data\ndem = /stale/dem\n")
    _write_auxiliary_sentinels(project_dir)
    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )

    project = notebook_setup.prepare_report_project(project_dir)

    config = ConfigParser()
    config.read(project.config_path)
    assert "dem" not in config["path"]


def test_prepare_report_project_rejects_file_dem_path(tmp_path):
    dem_file = tmp_path / "dem.tif"
    dem_file.write_text("not a directory")

    with pytest.raises(NotADirectoryError, match="DEM path should be a directory"):
        notebook_setup.prepare_report_project(
            tmp_path / "reports",
            skip_aux_download=True,
            dem_path=dem_file,
        )


def test_main_uses_test_dem_env(monkeypatch, tmp_path):
    report_project = tmp_path / "reports"
    dem_dir = tmp_path / "external-dem"
    dem_dir.mkdir()
    _write_auxiliary_sentinels(report_project)
    monkeypatch.setenv(notebook_setup.TEST_DEM_ENV_VAR, str(dem_dir))
    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )

    assert (
        notebook_setup.main(["reports", "--report-project", str(report_project)]) == 0
    )

    config = ConfigParser()
    config.read(report_project / "cryoswath.cfg")
    assert config["path"]["dem"] == str(dem_dir.resolve())


def test_main_uses_external_test_data_without_downloading(monkeypatch, tmp_path):
    report_project = tmp_path / "reports"
    data_dir = tmp_path / "data"
    _write_auxiliary_sentinels(tmp_path)
    (data_dir / "L1b").mkdir(parents=True)
    monkeypatch.setenv(notebook_setup.TEST_DATA_ENV_VAR, str(data_dir))
    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )

    assert (
        notebook_setup.main(["reports", "--report-project", str(report_project)]) == 0
    )

    config = ConfigParser()
    config.read(report_project / "cryoswath.cfg")
    assert config["path"]["data"] == str(data_dir.resolve())


def test_prepare_report_project_reuses_existing_auxiliary(monkeypatch, tmp_path):
    _write_auxiliary_sentinels(tmp_path / "reports")
    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )

    project = notebook_setup.prepare_report_project(tmp_path / "reports")

    assert project.config_path == tmp_path / "reports" / "cryoswath.cfg"


def test_prepare_tutorial_project_copies_resources_and_support_files(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "repo"
    source_data = repo_root / "data" / "tutorials"
    source_data.mkdir(parents=True)
    for filename in notebook_setup.TUTORIAL_SUPPORT_FILES:
        (source_data / filename).write_text(filename)

    def fake_download_auxiliary_data(base_dir=".", *, force=False, timeout=120):
        _write_auxiliary_sentinels(Path(base_dir))
        return str(Path(base_dir) / "data" / "auxiliary")

    def fake_copy_tutorials(destination=None, *, base_dir=".", force=False):
        tutorial_dir = Path(base_dir) / "tutorials"
        tutorial_dir.mkdir(parents=True)
        (tutorial_dir / "tutorial__example.ipynb").write_text("{}")
        return str(tutorial_dir)

    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        fake_download_auxiliary_data,
    )
    monkeypatch.setattr(notebook_setup.misc, "copy_tutorials", fake_copy_tutorials)

    project = notebook_setup.prepare_tutorial_project(
        tmp_path / "tutorial-project",
        repo_root=repo_root,
    )

    assert (project.tutorial_dir / "tutorial__example.ipynb").is_file()
    assert (
        tmp_path
        / "tutorial-project"
        / "data"
        / "auxiliary"
        / "DEM"
        / "arcticdem_mosaic_100m_v4.1_dem__excerpt_barnes-ice-cap.tif"
    ).is_file()
    assert (
        tmp_path
        / "tutorial-project"
        / "data"
        / "auxiliary"
        / "RGI"
        / "barnes_ice_cap.feather"
    ).is_file()
    assert (
        tmp_path / "tutorial-project" / "data" / "tutorials" / "barnes_ice_cap.feather"
    ).is_file()


def test_prepare_tutorial_project_uses_external_data_support(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data"
    _write_auxiliary_sentinels(tmp_path)
    for filename in notebook_setup.TUTORIAL_SUPPORT_FILES:
        source = data_dir / "tutorials" / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(filename)

    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )

    def fake_copy_tutorials(destination=None, *, base_dir=".", force=False):
        tutorial_dir = Path(base_dir) / "tutorials"
        tutorial_dir.mkdir(parents=True)
        return str(tutorial_dir)

    monkeypatch.setattr(notebook_setup.misc, "copy_tutorials", fake_copy_tutorials)

    project = notebook_setup.prepare_tutorial_project(
        tmp_path / "tutorial-project",
        repo_root=repo_root,
        data_path=data_dir,
    )

    for destinations in notebook_setup.TUTORIAL_SUPPORT_FILES.values():
        assert all(
            (project.project_dir / destination).is_file()
            for destination in destinations
        )


def test_prepare_tutorial_project_fails_early_for_missing_support_files(
    monkeypatch, tmp_path
):
    def fake_download_auxiliary_data(base_dir=".", *, force=False, timeout=120):
        _write_auxiliary_sentinels(Path(base_dir))
        return str(Path(base_dir) / "data" / "auxiliary")

    monkeypatch.setattr(
        notebook_setup.misc,
        "download_auxiliary_data",
        fake_download_auxiliary_data,
    )

    with pytest.raises(FileNotFoundError, match="Missing tutorial support"):
        notebook_setup.prepare_tutorial_project(
            tmp_path / "tutorial-project",
            repo_root=tmp_path / "repo",
        )
