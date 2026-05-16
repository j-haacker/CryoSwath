import sys

import pytest

import cryoswath.misc as misc


def _fail_if_called(*args, **kwargs):
    raise AssertionError("entry point work should not run for --help")


@pytest.mark.parametrize(
    ("command", "function_name"),
    [
        ("cryoswath-init", "init_project_cli"),
        ("cryoswath-download-rgi", "download_rgi_cli"),
        ("cryoswath-update-keyring", "update_keyring_cli"),
        ("cryoswath-update-netrc", "update_netrc_cli"),
        ("cryoswath-update-tracks", "update_track_database_cli"),
    ],
)
def test_legacy_console_entry_help_exits_before_work(
    command, function_name, monkeypatch, capsys
):
    monkeypatch.setattr(misc, "create_config", _fail_if_called)
    monkeypatch.setattr(misc, "download_rgi_o1region", _fail_if_called)
    monkeypatch.setattr(misc, "update_keyring", _fail_if_called)
    monkeypatch.setattr(misc, "update_netrc", _fail_if_called)
    monkeypatch.setattr(misc, "update_track_database", _fail_if_called)

    monkeypatch.setattr(sys, "argv", [command, "--help"])
    with pytest.raises(SystemExit) as excinfo:
        getattr(misc, function_name)()

    assert excinfo.value.code == 0
    assert f"usage: {command}" in capsys.readouterr().out


def test_cryoswath_help_exits_before_work(monkeypatch, capsys):
    monkeypatch.setattr(misc, "create_config", _fail_if_called)
    monkeypatch.setattr(misc, "download_auxiliary_data", _fail_if_called)
    monkeypatch.setattr(misc, "copy_tutorials", _fail_if_called)
    monkeypatch.setattr(misc, "download_rgi_o1region", _fail_if_called)
    monkeypatch.setattr(misc, "update_track_database", _fail_if_called)

    monkeypatch.setattr(sys, "argv", ["cryoswath", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        misc.cryoswath_cli()

    assert excinfo.value.code == 0
    assert "usage: cryoswath" in capsys.readouterr().out


@pytest.mark.parametrize(
    "subcommand",
    [
        "create-config",
        "download-aux-data",
        "get-tutorials",
        "download-rgi",
        "update-tracks",
        "update-keyring",
        "update-netrc",
    ],
)
def test_cryoswath_subcommand_help_exits_before_work(
    subcommand, monkeypatch, capsys
):
    monkeypatch.setattr(misc, "create_config", _fail_if_called)
    monkeypatch.setattr(misc, "download_auxiliary_data", _fail_if_called)
    monkeypatch.setattr(misc, "copy_tutorials", _fail_if_called)
    monkeypatch.setattr(misc, "download_rgi_o1region", _fail_if_called)
    monkeypatch.setattr(misc, "update_track_database", _fail_if_called)

    monkeypatch.setattr(sys, "argv", ["cryoswath", subcommand, "--help"])
    with pytest.raises(SystemExit) as excinfo:
        misc.cryoswath_cli()

    assert excinfo.value.code == 0
    assert f"usage: cryoswath {subcommand}" in capsys.readouterr().out


def test_init_project_cli_dispatches_after_parsing(monkeypatch):
    observed = {}

    def fake_create_config(
        config_file="cryoswath.cfg", data="data", *, base_dir=".", force=False
    ):
        observed["config_file"] = config_file
        observed["data"] = data
        observed["base_dir"] = base_dir
        observed["force"] = force
        return "custom.cfg"

    monkeypatch.setattr(misc, "create_config", fake_create_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath-init",
            "--base-dir",
            "project",
            "--config",
            "custom.cfg",
            "--data",
            "custom-data",
            "--force",
        ],
    )

    misc.init_project_cli()

    assert observed == {
        "config_file": "custom.cfg",
        "data": "custom-data",
        "base_dir": "project",
        "force": True,
    }


def test_cryoswath_create_config_dispatches_after_parsing(monkeypatch):
    observed = {}

    def fake_create_config(config_file, data, *, base_dir, force):
        observed.update(
            config_file=config_file, data=data, base_dir=base_dir, force=force
        )
        return "project/cryoswath.cfg"

    monkeypatch.setattr(misc, "create_config", fake_create_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath",
            "create-config",
            "--base-dir",
            "project",
            "--config",
            "conf.cfg",
            "--data",
            "store",
            "--force",
        ],
    )

    misc.cryoswath_cli()

    assert observed == {
        "config_file": "conf.cfg",
        "data": "store",
        "base_dir": "project",
        "force": True,
    }


def test_cryoswath_download_aux_data_dispatches_after_parsing(monkeypatch, capsys):
    observed = {}

    def fake_download_auxiliary_data(base_dir=".", *, force=False, timeout=120):
        observed.update(base_dir=base_dir, force=force, timeout=timeout)
        return "/tmp/project/data/auxiliary"

    monkeypatch.setattr(misc, "download_auxiliary_data", fake_download_auxiliary_data)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath",
            "download-aux-data",
            "--base-dir",
            "project",
            "--force",
            "--timeout",
            "9",
        ],
    )

    misc.cryoswath_cli()

    assert observed == {"base_dir": "project", "force": True, "timeout": 9.0}
    assert "/tmp/project/data/auxiliary" in capsys.readouterr().out


def test_cryoswath_get_tutorials_dispatches_after_parsing(monkeypatch, capsys):
    observed = {}

    def fake_copy_tutorials(destination=None, *, base_dir=".", force=False):
        observed.update(destination=destination, base_dir=base_dir, force=force)
        return "/tmp/project/tutorials"

    monkeypatch.setattr(misc, "copy_tutorials", fake_copy_tutorials)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath",
            "get-tutorials",
            "--base-dir",
            "project",
            "--destination",
            "notebooks",
            "--force",
        ],
    )

    misc.cryoswath_cli()

    assert observed == {
        "destination": "notebooks",
        "base_dir": "project",
        "force": True,
    }
    assert "/tmp/project/tutorials" in capsys.readouterr().out


def test_cryoswath_download_rgi_dispatches_after_parsing(monkeypatch, capsys):
    observed = {}

    def fake_download_rgi_o1region(o1code, product, force, timeout):
        observed.update(o1code=o1code, product=product, force=force, timeout=timeout)
        return "/tmp/rgi"

    monkeypatch.setattr(misc, "download_rgi_o1region", fake_download_rgi_o1region)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath",
            "download-rgi",
            "--o1",
            "09",
            "--product",
            "complexes",
            "--force",
            "--timeout",
            "7",
        ],
    )

    misc.cryoswath_cli()

    assert observed == {
        "o1code": "09",
        "product": "complexes",
        "force": True,
        "timeout": 7.0,
    }
    assert "/tmp/rgi" in capsys.readouterr().out


def test_cryoswath_update_track_database_dispatches_after_parsing(monkeypatch):
    calls = []

    monkeypatch.setattr(
        misc, "update_track_database", lambda: calls.append("update-tracks")
    )
    monkeypatch.setattr(sys, "argv", ["cryoswath", "update-tracks"])

    misc.cryoswath_cli()

    assert calls == ["update-tracks"]


def test_cryoswath_update_keyring_dispatches_after_parsing(monkeypatch, capsys):
    observed = {}

    def fake_update_keyring(user, password, *, service, username_key):
        observed.update(
            user=user, password=password, service=service, username_key=username_key
        )
        return "esa-user"

    monkeypatch.setattr(misc, "update_keyring", fake_update_keyring)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath",
            "update-keyring",
            "--user",
            "u",
            "--password",
            "p",
            "--service",
            "svc",
            "--username-key",
            "default",
        ],
    )

    misc.cryoswath_cli()

    assert observed == {
        "user": "u",
        "password": "p",
        "service": "svc",
        "username_key": "default",
    }
    assert "Stored credentials for esa-user" in capsys.readouterr().out


def test_cryoswath_update_netrc_dispatches_after_parsing(monkeypatch, capsys):
    observed = {}

    def fake_update_netrc(user, password, *, machine, netrc_file):
        observed.update(
            user=user, password=password, machine=machine, netrc_file=netrc_file
        )
        return "/tmp/netrc"

    monkeypatch.setattr(misc, "update_netrc", fake_update_netrc)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cryoswath",
            "update-netrc",
            "--user",
            "u",
            "--password",
            "p",
            "--machine",
            "machine",
            "--netrc-file",
            "netrc",
        ],
    )

    misc.cryoswath_cli()

    assert observed == {
        "user": "u",
        "password": "p",
        "machine": "machine",
        "netrc_file": "netrc",
    }
    assert "Wrote plaintext credentials" in capsys.readouterr().out


def test_update_track_database_cli_dispatches_after_parsing(monkeypatch):
    calls = []

    monkeypatch.setattr(
        misc, "update_track_database", lambda: calls.append("update-tracks")
    )
    monkeypatch.setattr(sys, "argv", ["cryoswath-update-tracks"])

    misc.update_track_database_cli()

    assert calls == ["update-tracks"]
