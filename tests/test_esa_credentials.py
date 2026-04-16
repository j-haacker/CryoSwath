import pytest

import cryoswath.misc as misc


def test_resolve_esa_credentials_prefers_environment_variables(monkeypatch):
    monkeypatch.setenv("EOIAM_USER", "env-user")
    monkeypatch.setenv("EOIAM_PASSWORD", "env-password")
    monkeypatch.setattr(
        misc,
        "_resolve_esa_keyring_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("keyring should not be used")),
    )
    user, password, source = misc._resolve_esa_ftp_credentials()
    assert user == "env-user"
    assert password == "env-password"
    assert source == "environment variables"


def test_resolve_esa_credentials_uses_keyring_before_netrc(monkeypatch):
    monkeypatch.delenv("EOIAM_USER", raising=False)
    monkeypatch.delenv("EOIAM_PASSWORD", raising=False)
    monkeypatch.setattr(
        misc,
        "_resolve_esa_keyring_credentials",
        lambda: ("keyring-user", "keyring-password", "keyring"),
    )

    class NetrcNotExpected:
        def authenticators(self, machine):
            raise AssertionError("netrc should not be used when keyring is available")

    monkeypatch.setattr(misc.netrc, "netrc", lambda: NetrcNotExpected())
    user, password, source = misc._resolve_esa_ftp_credentials()
    assert user == "keyring-user"
    assert password == "keyring-password"
    assert source == "keyring"


def test_resolve_esa_credentials_uses_netrc_when_keyring_missing(monkeypatch):
    monkeypatch.delenv("EOIAM_USER", raising=False)
    monkeypatch.delenv("EOIAM_PASSWORD", raising=False)
    monkeypatch.setattr(misc, "_resolve_esa_keyring_credentials", lambda: None)

    class FakeNetrc:
        def authenticators(self, machine):
            assert machine == misc._ESA_CS2_HOST
            return ("netrc-user", None, "netrc-password")

    monkeypatch.setattr(misc.netrc, "netrc", lambda: FakeNetrc())
    user, password, source = misc._resolve_esa_ftp_credentials()
    assert user == "netrc-user"
    assert password == "netrc-password"
    assert source == "~/.netrc"


def test_resolve_esa_credentials_ignores_obsolete_cryoswath_ftp_env_vars(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EOIAM_USER", raising=False)
    monkeypatch.delenv("EOIAM_PASSWORD", raising=False)
    monkeypatch.setenv("CRYOSWATH_FTP_USER", "legacy-user")
    monkeypatch.setenv("CRYOSWATH_FTP_PASSWORD", "legacy-password")
    monkeypatch.setattr(misc, "_resolve_esa_keyring_credentials", lambda: None)

    def _missing_netrc():
        raise FileNotFoundError()

    monkeypatch.setattr(misc.netrc, "netrc", _missing_netrc)

    with pytest.raises(RuntimeError, match="No ESA credentials found"):
        misc._resolve_esa_ftp_credentials()


def test_update_keyring_stores_and_verifies(monkeypatch):
    store = {}

    class FakeKeyring:
        def set_password(self, service, user, password):
            store[(service, user)] = password

        def get_password(self, service, user):
            return store.get((service, user))

    monkeypatch.setattr(misc, "keyring", FakeKeyring())
    user = misc.update_keyring(user="esa-user", password="esa-password")
    assert user == "esa-user"
    assert store[(misc._ESA_AUTH_IDP_HOST, "esa-user")] == "esa-password"
    assert (
        store[(misc._ESA_AUTH_IDP_HOST, misc._ESA_KEYRING_DEFAULT_USER_KEY)]
        == "esa-user"
    )


def test_update_keyring_raises_for_backend_errors(monkeypatch):
    class FailingKeyring:
        def set_password(self, service, user, password):
            raise misc.KeyringError("backend down")

    monkeypatch.setattr(misc, "keyring", FailingKeyring())
    with pytest.raises(RuntimeError, match="backend down"):
        misc.update_keyring(user="esa-user", password="esa-password")
