import sys

import pytest

import cryoswath.misc as misc


def _fail_if_called():
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
def test_console_entry_help_exits_before_work(
    command, function_name, monkeypatch, capsys
):
    if function_name == "init_project_cli":
        monkeypatch.setattr(misc, "init_project", _fail_if_called)
    elif function_name == "update_track_database_cli":
        monkeypatch.setattr(misc, "update_track_database", _fail_if_called)

    monkeypatch.setattr(sys, "argv", [command, "--help"])
    with pytest.raises(SystemExit) as excinfo:
        getattr(misc, function_name)()

    assert excinfo.value.code == 0
    assert f"usage: {command}" in capsys.readouterr().out


def test_init_project_cli_dispatches_after_parsing(monkeypatch):
    calls = []

    monkeypatch.setattr(misc, "init_project", lambda: calls.append("init"))
    monkeypatch.setattr(sys, "argv", ["cryoswath-init"])

    misc.init_project_cli()

    assert calls == ["init"]


def test_update_track_database_cli_dispatches_after_parsing(monkeypatch):
    calls = []

    monkeypatch.setattr(
        misc, "update_track_database", lambda: calls.append("update-tracks")
    )
    monkeypatch.setattr(sys, "argv", ["cryoswath-update-tracks"])

    misc.update_track_database_cli()

    assert calls == ["update-tracks"]
