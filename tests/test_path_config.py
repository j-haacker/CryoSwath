from configparser import ConfigParser
from pathlib import Path

import pytest

import cryoswath.misc as misc


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {"XDG_CONFIG_HOME": str(tmp_path / "empty-xdg")}
    env.update(overrides)
    return env


def _write_config(path: Path, data: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = ConfigParser()
    config["path"] = data
    with open(path, "w") as f:
        config.write(f)
    return path


def _resolve(cwd: Path, environ: dict[str, str]):
    return misc._resolve_path_configuration(cwd=cwd, environ=environ)


def test_path_config_falls_back_to_cwd_data(tmp_path):
    work = tmp_path / "work"
    work.mkdir()

    config, config_file, paths = _resolve(work, _env(tmp_path))

    assert config_file is None
    assert "path" in config
    assert paths["data"] == work / "data"
    assert paths["l1b"] == work / "data" / "L1b"
    assert paths["aux"] == work / "data" / "auxiliary"
    assert paths["dem"] == work / "data" / "auxiliary" / "DEM"


def test_path_config_discovers_cryoswath_cfg_in_parent(tmp_path):
    project = tmp_path / "project"
    child = project / "notebooks"
    child.mkdir(parents=True)
    config_file = _write_config(
        project / "cryoswath.cfg",
        {
            "data": "store",
            "l3": "products/L3",
            "aux": str(tmp_path / "shared-aux"),
            "dem": "dem-cache",
        },
    )

    _, resolved_config, paths = _resolve(child, _env(tmp_path))

    assert resolved_config == config_file
    assert paths["data"] == project / "store"
    assert paths["l3"] == project / "store" / "products" / "L3"
    assert paths["aux"] == tmp_path / "shared-aux"
    assert paths["dem"] == tmp_path / "shared-aux" / "dem-cache"


def test_path_config_environment_overrides_config(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write_config(
        project / "cryoswath.cfg",
        {"data": "config-data", "l4": "config-l4", "rgi": "config-rgi"},
    )

    _, _, paths = _resolve(
        project,
        _env(
            tmp_path,
            CRYOSWATH_DATA=str(tmp_path / "env-data"),
            CRYOSWATH_L4="env-l4",
            CRYOSWATH_RGI=str(tmp_path / "env-rgi"),
        ),
    )

    assert paths["data"] == tmp_path / "env-data"
    assert paths["l4"] == tmp_path / "env-data" / "env-l4"
    assert paths["rgi"] == tmp_path / "env-rgi"


def test_path_config_explicit_file_selection(tmp_path):
    project = tmp_path / "project"
    child = project / "child"
    child.mkdir(parents=True)
    _write_config(project / "cryoswath.cfg", {"data": "project-data"})
    explicit_config = _write_config(
        tmp_path / "selected.cfg", {"data": "selected-data"}
    )

    _, resolved_config, paths = _resolve(
        child,
        _env(tmp_path, CRYOSWATH_CONFIG=str(explicit_config)),
    )

    assert resolved_config == explicit_config
    assert paths["data"] == tmp_path / "selected-data"


@pytest.mark.parametrize(
    "relative_config",
    [Path("config.ini"), Path("scripts/config.ini")],
)
def test_path_config_discovers_legacy_config_ini(tmp_path, relative_config):
    project = tmp_path / "project"
    child = project / "work"
    child.mkdir(parents=True)
    config_file = _write_config(project / relative_config, {"data": "legacy-data"})

    _, resolved_config, paths = _resolve(child, _env(tmp_path))

    assert resolved_config == config_file
    assert paths["data"] == project / "legacy-data"


def test_path_config_prefers_new_config_over_legacy_config(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write_config(project / "config.ini", {"data": "legacy-data"})
    new_config = _write_config(project / "cryoswath.cfg", {"data": "new-data"})

    _, resolved_config, paths = _resolve(project, _env(tmp_path))

    assert resolved_config == new_config
    assert paths["data"] == project / "new-data"


def test_path_config_uses_user_config_as_last_config_source(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    xdg_config = _write_config(
        tmp_path / "xdg" / "cryoswath" / "cryoswath.cfg",
        {"data": str(tmp_path / "user-data")},
    )

    _, resolved_config, paths = _resolve(
        work, {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    )

    assert resolved_config == xdg_config
    assert paths["data"] == tmp_path / "user-data"


def test_create_config_writes_new_config_without_branch_cloning(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    out = misc.create_config()

    assert out == str(tmp_path / "cryoswath.cfg")
    config = ConfigParser()
    config.read(tmp_path / "cryoswath.cfg")
    assert config["path"]["data"] == "data"
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "scripts").exists()


def test_create_config_allows_existing_data_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    misc.create_config()

    assert (tmp_path / "cryoswath.cfg").is_file()
    assert (tmp_path / "data").is_dir()


def test_init_project_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "cryoswath.cfg").write_text("[path]\ndata = old\n")

    with pytest.raises(FileExistsError, match="--force"):
        misc.init_project()


def test_init_project_overwrites_with_force_and_preserves_other_sections(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "cryoswath.cfg").write_text(
        "[path]\ndata = old\n\n[defaults.fill_voids]\noutlier_limit = 4\n"
    )

    misc.init_project(data="new-data", force=True)

    config = ConfigParser()
    config.read(tmp_path / "cryoswath.cfg")
    assert config["path"]["data"] == "new-data"
    assert config["defaults.fill_voids"]["outlier_limit"] == "4"


def test_init_project_honors_custom_config_and_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    misc.init_project(config_file="conf/cryoswath.cfg", data="store")

    config = ConfigParser()
    config.read(tmp_path / "conf" / "cryoswath.cfg")
    assert config["path"]["data"] == "store"


def test_create_config_honors_base_dir_and_child_discovery(tmp_path):
    project = tmp_path / "project"

    out = misc.create_config(base_dir=project, data="store")

    assert out == str(project / "cryoswath.cfg")
    child = project / "tutorials"
    child.mkdir(parents=True)

    _, resolved_config, paths = _resolve(child, _env(tmp_path))

    assert resolved_config == project / "cryoswath.cfg"
    assert paths["data"] == project / "store"


def test_legacy_credentials_are_read_from_discovered_config_ini(monkeypatch, tmp_path):
    project = tmp_path / "project"
    scripts = project / "scripts"
    work = project / "notebooks"
    scripts.mkdir(parents=True)
    work.mkdir()
    (scripts / "config.ini").write_text(
        "[user]\nname = legacy-user\npassword = legacy-password\n"
    )
    monkeypatch.chdir(work)
    monkeypatch.delenv("EOIAM_USER", raising=False)
    monkeypatch.delenv("EOIAM_PASSWORD", raising=False)
    monkeypatch.setattr(misc, "_resolve_esa_keyring_credentials", lambda: None)

    def missing_netrc():
        raise FileNotFoundError()

    monkeypatch.setattr(misc.netrc, "netrc", missing_netrc)

    with pytest.warns(DeprecationWarning, match="config.ini"):
        user, password, source = misc._resolve_esa_ftp_credentials()

    assert (user, password, source) == (
        "legacy-user",
        "legacy-password",
        "config.ini [user] name/password",
    )
