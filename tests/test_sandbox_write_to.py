from pathlib import Path

import pytest

from cryoswath.misc import sandbox_write_to


def test_stale_backup_is_removed_when_target_exists(tmp_path: Path):
    target = tmp_path / "cache.h5"
    backup = tmp_path / "cache.h5__backup"
    target.write_text("current")
    backup.write_text("backup")

    with sandbox_write_to(str(target)):
        assert target.exists()
        assert not backup.exists()
        backup.write_text("new backup")

    assert target.exists()
    assert not backup.exists()


def test_backup_recovers_missing_target(tmp_path: Path):
    target = tmp_path / "cache.h5"
    backup = tmp_path / "cache.h5__backup"
    backup.write_text("recovered")

    with sandbox_write_to(str(target)):
        assert target.exists()
        assert target.read_text() == "recovered"


def test_lock_blocks_concurrent_writer(tmp_path: Path):
    target = tmp_path / "cache.h5"

    with sandbox_write_to(str(target)):
        with pytest.raises(Exception, match="Write lock"):
            with sandbox_write_to(str(target)):
                pass
