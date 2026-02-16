from pathlib import Path
import stat

from cryoswath.misc import update_netrc


def test_update_netrc_creates_entry(tmp_path: Path):
    netrc_path = tmp_path / ".netrc"
    written = update_netrc(
        user="esa-user",
        password="esa-password",
        machine="science-pds.cryosat.esa.int",
        netrc_file=netrc_path,
    )
    assert Path(written) == netrc_path.resolve()
    text = netrc_path.read_text()
    assert "machine science-pds.cryosat.esa.int" in text
    assert "login esa-user" in text
    assert "password esa-password" in text
    assert stat.S_IMODE(netrc_path.stat().st_mode) == 0o600


def test_update_netrc_replaces_machine_block(tmp_path: Path):
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(
        "machine science-pds.cryosat.esa.int\n"
        "  login old-user\n"
        "  password old-password\n\n"
        "machine other.example.org\n"
        "  login keep-user\n"
        "  password keep-password\n"
    )
    update_netrc(
        user="new-user",
        password="new-password",
        machine="science-pds.cryosat.esa.int",
        netrc_file=netrc_path,
    )
    text = netrc_path.read_text()
    assert "login old-user" not in text
    assert "password old-password" not in text
    assert "login new-user" in text
    assert "password new-password" in text
    assert "machine other.example.org" in text
    assert "login keep-user" in text
