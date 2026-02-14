from cryoswath.l1b import _status


def test_status_message_prefix(capsys):
    _status("example message")
    assert capsys.readouterr().out.strip() == "[l1b] example message"
