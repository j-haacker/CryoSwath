import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest

import cryoswath.misc as misc


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _clear_path_env(monkeypatch, tmp_path):
    monkeypatch.delenv("CRYOSWATH_CONFIG", raising=False)
    monkeypatch.delenv("CRYOSWATH_DATA", raising=False)
    monkeypatch.delenv("CRYOSWATH_AUX", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _write_zip(path: Path, entries: dict[str, str | bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(zip_path: Path, url="https://zenodo.example/CryoSwath-aux-data.zip"):
    return {
        "files": [
            {
                "key": "CryoSwath-aux-data.zip",
                "checksum": f"md5:{_md5(zip_path)}",
                "links": {"self": url},
            }
        ]
    }


def _mock_zenodo(monkeypatch, zip_path: Path):
    calls = {"metadata": [], "download": []}

    def fake_get(url, timeout=120):
        calls["metadata"].append((url, timeout))
        return FakeResponse(_metadata(zip_path))

    def fake_download_file(url, dest, timeout=120):
        calls["download"].append((url, Path(dest), timeout))
        shutil.copyfile(zip_path, dest)
        return str(dest)

    monkeypatch.setattr(misc.requests, "get", fake_get)
    monkeypatch.setattr(misc, "download_file", fake_download_file)
    return calls


def test_download_auxiliary_data_uses_zenodo_metadata_and_extracts(
    monkeypatch, tmp_path
):
    _clear_path_env(monkeypatch, tmp_path)
    zip_path = _write_zip(
        tmp_path / "aux.zip",
        {
            "CryoSat-2_SARIn_ground_tracks.feather": b"tracks",
            "CryoSat-2_SARIn_file_names.pkl": b"names",
            "RGI/RGI2000-v7.0-o1regions.feather": b"rgi",
        },
    )
    calls = _mock_zenodo(monkeypatch, zip_path)

    out = misc.download_auxiliary_data(base_dir=tmp_path, timeout=7)

    aux_dir = tmp_path / "data" / "auxiliary"
    assert out == str(aux_dir)
    assert (aux_dir / "CryoSat-2_SARIn_ground_tracks.feather").read_bytes() == b"tracks"
    assert (aux_dir / "CryoSat-2_SARIn_file_names.pkl").read_bytes() == b"names"
    assert (aux_dir / "RGI" / "RGI2000-v7.0-o1regions.feather").read_bytes() == b"rgi"
    assert calls["metadata"] == [(misc._ZENODO_AUX_CONCEPT_RECORD_API_URL, 7)]
    assert calls["download"][0][0] == "https://zenodo.example/CryoSwath-aux-data.zip"
    assert calls["download"][0][2] == 7
    assert not list(aux_dir.glob("*.zip"))


def test_download_auxiliary_data_preserves_existing_files_without_force(
    monkeypatch, tmp_path
):
    _clear_path_env(monkeypatch, tmp_path)
    aux_dir = tmp_path / "data" / "auxiliary"
    aux_dir.mkdir(parents=True)
    existing = aux_dir / "CryoSat-2_SARIn_ground_tracks.feather"
    existing.write_bytes(b"local-update")
    zip_path = _write_zip(
        tmp_path / "aux.zip",
        {
            "CryoSat-2_SARIn_ground_tracks.feather": b"zenodo-baseline",
            "RGI/new.txt": b"new",
        },
    )
    _mock_zenodo(monkeypatch, zip_path)

    misc.download_auxiliary_data(base_dir=tmp_path)

    assert existing.read_bytes() == b"local-update"
    assert (aux_dir / "RGI" / "new.txt").read_bytes() == b"new"


def test_download_auxiliary_data_force_replaces_archive_files(monkeypatch, tmp_path):
    _clear_path_env(monkeypatch, tmp_path)
    aux_dir = tmp_path / "data" / "auxiliary"
    aux_dir.mkdir(parents=True)
    existing = aux_dir / "CryoSat-2_SARIn_ground_tracks.feather"
    existing.write_bytes(b"local-update")
    zip_path = _write_zip(
        tmp_path / "aux.zip",
        {"CryoSat-2_SARIn_ground_tracks.feather": b"zenodo-baseline"},
    )
    _mock_zenodo(monkeypatch, zip_path)

    misc.download_auxiliary_data(base_dir=tmp_path, force=True)

    assert existing.read_bytes() == b"zenodo-baseline"


def test_download_auxiliary_data_rejects_bad_checksum(monkeypatch, tmp_path):
    _clear_path_env(monkeypatch, tmp_path)
    zip_path = _write_zip(tmp_path / "aux.zip", {"file.txt": b"payload"})

    def fake_get(url, timeout=120):
        return FakeResponse(
            {
                "files": [
                    {
                        "key": "CryoSwath-aux-data.zip",
                        "checksum": "md5:00000000000000000000000000000000",
                        "links": {"self": "https://zenodo.example/file.zip"},
                    }
                ]
            }
        )

    monkeypatch.setattr(misc.requests, "get", fake_get)

    def copy_zip(url, dest, timeout=120):
        return shutil.copyfile(zip_path, dest)

    monkeypatch.setattr(misc, "download_file", copy_zip)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        misc.download_auxiliary_data(base_dir=tmp_path)

    assert not (tmp_path / "data" / "auxiliary" / "file.txt").exists()


def test_download_auxiliary_data_rejects_non_zip_payload(monkeypatch, tmp_path):
    _clear_path_env(monkeypatch, tmp_path)
    payload = tmp_path / "not.zip"
    payload.write_text("not a zip")
    _mock_zenodo(monkeypatch, payload)

    with pytest.raises(RuntimeError, match="not a zip"):
        misc.download_auxiliary_data(base_dir=tmp_path)


def test_download_auxiliary_data_rejects_path_traversal(monkeypatch, tmp_path):
    _clear_path_env(monkeypatch, tmp_path)
    zip_path = _write_zip(tmp_path / "aux.zip", {"../evil.txt": b"bad"})
    _mock_zenodo(monkeypatch, zip_path)

    with pytest.raises(RuntimeError, match="Unsafe path"):
        misc.download_auxiliary_data(base_dir=tmp_path)

    assert not (tmp_path / "evil.txt").exists()


def test_download_auxiliary_data_cleans_temp_archive_after_download_failure(
    monkeypatch, tmp_path
):
    _clear_path_env(monkeypatch, tmp_path)
    zip_path = _write_zip(tmp_path / "aux.zip", {"file.txt": b"payload"})

    def metadata_response(url, timeout=120):
        return FakeResponse(_metadata(zip_path))

    monkeypatch.setattr(misc.requests, "get", metadata_response)

    def fail_download(url, dest, timeout=120):
        Path(dest).write_bytes(b"partial")
        raise RuntimeError("network failed")

    monkeypatch.setattr(misc, "download_file", fail_download)

    with pytest.raises(RuntimeError, match="network failed"):
        misc.download_auxiliary_data(base_dir=tmp_path)

    aux_dir = tmp_path / "data" / "auxiliary"
    assert not list(aux_dir.glob("*.zip"))


def test_packaged_tutorial_resources_are_discoverable():
    names = [resource.name for resource in misc._tutorial_resources()]

    assert names == [
        "tutorial__diagnostic_hooks.ipynb",
        "tutorial__general_step-by-step.ipynb",
        "tutorial__poca.ipynb",
        "tutorial__process_first_swath.ipynb",
        "tutorial__process_first_waveform.ipynb",
    ]


def test_copy_tutorials_defaults_to_base_tutorials(tmp_path):
    out = misc.copy_tutorials(base_dir=tmp_path)

    tutorial_dir = tmp_path / "tutorials"
    assert out == str(tutorial_dir)
    assert (tutorial_dir / "tutorial__diagnostic_hooks.ipynb").is_file()
    assert (tutorial_dir / "tutorial__general_step-by-step.ipynb").is_file()
    assert (tutorial_dir / "tutorial__process_first_waveform.ipynb").is_file()


def test_copy_tutorials_uses_custom_destination(tmp_path):
    out = misc.copy_tutorials(destination="notebooks", base_dir=tmp_path)

    assert out == str(tmp_path / "notebooks")
    assert (tmp_path / "notebooks" / "tutorial__poca.ipynb").is_file()


def test_copy_tutorials_refuses_existing_files_without_force(tmp_path):
    tutorial_dir = tmp_path / "tutorials"
    tutorial_dir.mkdir()
    existing = tutorial_dir / "tutorial__poca.ipynb"
    existing.write_text("local notes")

    with pytest.raises(FileExistsError, match="--force"):
        misc.copy_tutorials(base_dir=tmp_path)

    assert existing.read_text() == "local notes"


def test_copy_tutorials_force_overwrites_existing_files(tmp_path):
    tutorial_dir = tmp_path / "tutorials"
    tutorial_dir.mkdir()
    existing = tutorial_dir / "tutorial__poca.ipynb"
    existing.write_text("local notes")

    misc.copy_tutorials(base_dir=tmp_path, force=True)

    assert existing.read_text() != "local notes"
