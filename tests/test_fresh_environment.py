import pytest

fresh = pytest.importorskip("tools.test_fresh_environment")


def test_fresh_environment_passes_credentials_and_blocks_local_paths(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("EOIAM_USER", "esa-user")
    monkeypatch.setenv("EOIAM_PASSWORD", "esa-password")
    monkeypatch.setenv("EARTHDATA_USERNAME", "earth-user")
    monkeypatch.setenv("CRYOSWATH_DATA", "/local/data")
    monkeypatch.setenv("PYTHONPATH", "/local/source")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example")

    env = fresh.fresh_environment(tmp_path, pass_env=[])

    assert env["PATH"] == "/usr/bin"
    assert env["EOIAM_USER"] == "esa-user"
    assert env["EOIAM_PASSWORD"] == "esa-password"
    assert env["EARTHDATA_USERNAME"] == "earth-user"
    assert env["HTTPS_PROXY"] == "http://proxy.example"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["XDG_CONFIG_HOME"] == str(tmp_path / "home" / ".config")
    assert "CRYOSWATH_DATA" not in env
    assert "PYTHONPATH" not in env


def test_fresh_environment_allows_explicit_extra_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("CUSTOM_TOKEN", "secret")
    monkeypatch.setenv("CRYOSWATH_CONFIG", "/intentional/config")

    env = fresh.fresh_environment(
        tmp_path,
        pass_env=["CUSTOM_TOKEN", "CRYOSWATH_CONFIG"],
    )

    assert env["CUSTOM_TOKEN"] == "secret"
    assert env["CRYOSWATH_CONFIG"] == "/intentional/config"


def test_copy_netrc_sets_private_permissions(tmp_path):
    source_home = tmp_path / "source-home"
    target_home = tmp_path / "target-home"
    source_home.mkdir()
    (source_home / ".netrc").write_text(
        "machine science-pds.cryosat.esa.int login user password pass\n"
    )

    fresh.copy_netrc(source_home, target_home)

    target = target_home / ".netrc"
    assert target.read_text().startswith("machine science-pds")
    assert target.stat().st_mode & 0o777 == 0o600


def test_copy_netrc_fails_when_source_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .netrc"):
        fresh.copy_netrc(tmp_path / "missing-home", tmp_path / "target-home")


def test_copy_tracked_worktree_copies_only_git_tracked_files(tmp_path):
    repo = tmp_path / "repo"
    checkout = tmp_path / "checkout"
    repo.mkdir()
    fresh.run(["git", "init"], cwd=repo)
    fresh.run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    fresh.run(["git", "config", "user.name", "Test User"], cwd=repo)
    (repo / "tracked.txt").write_text("tracked")
    (repo / "untracked.txt").write_text("untracked")
    fresh.run(["git", "add", "tracked.txt"], cwd=repo)
    fresh.run(["git", "commit", "-m", "add tracked"], cwd=repo)
    (repo / "tracked.txt").write_text("modified")

    fresh.copy_tracked_worktree(repo, checkout)

    assert (checkout / "tracked.txt").read_text() == "modified"
    assert not (checkout / "untracked.txt").exists()
