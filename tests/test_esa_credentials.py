import pytest

import cryoswath.misc as misc


@pytest.mark.parametrize(
    ("kwargs", "expected_timeout"),
    [({}, misc._ESA_FTP_TIMEOUT), ({"timeout": 30}, 30)],
)
def test_ftp_cs2_server_uses_timeout(monkeypatch, kwargs, expected_timeout):
    observed = {}

    class FakeFtp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, **kwargs):
            observed["login"] = kwargs

        def prot_p(self):
            observed["protected"] = True

    def ftp_tls(host, **kwargs):
        observed["host"] = host
        observed["kwargs"] = kwargs
        return FakeFtp()

    monkeypatch.setattr(
        misc, "_resolve_esa_ftp_credentials", lambda: ("u", "p", "test")
    )
    monkeypatch.setattr(misc.ftplib, "FTP_TLS", ftp_tls)

    with misc.ftp_cs2_server(**kwargs) as ftp:
        assert isinstance(ftp, FakeFtp)

    assert observed == {
        "host": misc._ESA_CS2_HOST,
        "kwargs": {"timeout": expected_timeout},
        "login": {"user": "u", "passwd": "p"},
        "protected": True,
    }


def test_ftp_cs2_server_reports_setup_timeout(monkeypatch):
    monkeypatch.setattr(
        misc, "_resolve_esa_ftp_credentials", lambda: ("u", "p", "test")
    )
    monkeypatch.setattr(
        misc.ftplib,
        "FTP_TLS",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    with pytest.raises(RuntimeError, match="timed out after 60 seconds"):
        with misc.ftp_cs2_server():
            pass


def test_ftp_cs2_server_reports_operation_timeout(monkeypatch):
    class FakeFtp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, **kwargs):
            pass

        def prot_p(self):
            pass

        def nlst(self):
            raise TimeoutError()

    monkeypatch.setattr(
        misc, "_resolve_esa_ftp_credentials", lambda: ("u", "p", "test")
    )
    monkeypatch.setattr(misc.ftplib, "FTP_TLS", lambda *args, **kwargs: FakeFtp())

    with pytest.raises(RuntimeError, match="timed out after 60 seconds"):
        with misc.ftp_cs2_server() as ftp:
            ftp.nlst()


def test_ftp_cs2_server_preserves_authentication_error(monkeypatch):
    class FakeFtp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, **kwargs):
            raise misc.ftplib.error_perm("denied")

    monkeypatch.setattr(
        misc, "_resolve_esa_ftp_credentials", lambda: ("u", "p", "test source")
    )
    monkeypatch.setattr(misc.ftplib, "FTP_TLS", lambda *args, **kwargs: FakeFtp())

    with pytest.raises(RuntimeError, match="authentication failed.*test source"):
        with misc.ftp_cs2_server():
            pass


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


def test_resolve_maap_token_prefers_environment(monkeypatch):
    monkeypatch.setenv("ESA_MAAP_OFFLINE_TOKEN", "environment-token")

    class KeyringNotExpected:
        def get_password(self, service, user):
            raise AssertionError("keyring should not be used")

    monkeypatch.setattr(misc, "keyring", KeyringNotExpected())

    token, source = misc._resolve_esa_maap_offline_token()

    assert token == "environment-token"
    assert source == "environment variable ESA_MAAP_OFFLINE_TOKEN"


def test_resolve_maap_token_uses_dedicated_keyring_entry(monkeypatch):
    class FakeKeyring:
        def get_password(self, service, user):
            assert service == misc._ESA_MAAP_KEYRING_SERVICE
            assert user == misc._ESA_MAAP_KEYRING_USER
            return "keyring-token"

    monkeypatch.delenv("ESA_MAAP_OFFLINE_TOKEN", raising=False)
    monkeypatch.setattr(misc, "keyring", FakeKeyring())

    assert misc._resolve_esa_maap_offline_token() == (
        "keyring-token",
        f"keyring service {misc._ESA_MAAP_KEYRING_SERVICE}",
    )


def test_update_maap_token_stores_and_verifies(monkeypatch):
    store = {}

    class FakeKeyring:
        def set_password(self, service, user, password):
            store[(service, user)] = password

        def get_password(self, service, user):
            return store.get((service, user))

    monkeypatch.setattr(misc, "keyring", FakeKeyring())

    assert misc.update_maap_token("offline-token") == "offline-token"
    assert store[(misc._ESA_MAAP_KEYRING_SERVICE, misc._ESA_MAAP_KEYRING_USER)] == (
        "offline-token"
    )


def test_resolve_maap_token_explains_configuration(monkeypatch):
    monkeypatch.delenv("ESA_MAAP_OFFLINE_TOKEN", raising=False)
    monkeypatch.setattr(misc, "keyring", None)

    with pytest.raises(RuntimeError, match="update-maap-token"):
        misc._resolve_esa_maap_offline_token()
