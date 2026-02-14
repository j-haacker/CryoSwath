import cryoswath.l2 as l2


def test_get_parallel_pool_uses_spawn_context(monkeypatch):
    expected_pool = object()
    called = {}

    class DummyContext:
        Pool = expected_pool

    def fake_get_context(method):
        called["method"] = method
        return DummyContext

    monkeypatch.setattr(l2.mp, "get_context", fake_get_context)
    assert l2._get_parallel_pool() is expected_pool
    assert called["method"] == "spawn"
