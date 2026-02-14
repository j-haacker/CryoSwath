import warnings

import cryoswath.l2 as l2


def test_detect_available_cores_prefers_affinity(monkeypatch):
    monkeypatch.setattr(l2.os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(l2.os, "cpu_count", lambda: 8)
    assert l2._detect_available_cores() == 3


def test_detect_available_cores_falls_back_to_cpu_count(monkeypatch):
    def _raise_attribute_error(_pid):
        raise AttributeError("no affinity support")

    monkeypatch.setattr(l2.os, "sched_getaffinity", _raise_attribute_error, raising=False)
    monkeypatch.setattr(l2.os, "cpu_count", lambda: 6)
    assert l2._detect_available_cores() == 6


def test_detect_available_cores_warns_and_defaults_to_one(monkeypatch):
    def _raise_not_implemented(_pid):
        raise NotImplementedError("unsupported")

    monkeypatch.setattr(l2.os, "sched_getaffinity", _raise_not_implemented, raising=False)
    monkeypatch.setattr(l2.os, "cpu_count", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cores = l2._detect_available_cores()
    assert cores == 1
    assert any("Failed to find number of CPU cores" in str(w.message) for w in caught)
