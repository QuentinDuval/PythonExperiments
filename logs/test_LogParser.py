

from logs.LogParser import *
import pytest


def test_tokenizer():
    tokenizer = Tokenizer("abc  [fg h] \"i j\" kl")
    assert "abc" == tokenizer.take_until(" ")
    assert "fg h" == tokenizer.take_between("[", "]")
    assert "i j" == tokenizer.take_between("\"", "\"")
    assert "kl" == tokenizer.remaining()


def test_log_entry_parsing():
    log_line = '127.0.0.1 - mary [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 503 12'
    expected_log = LogEntry(
        remote_host_name='127.0.0.1',
        auth_user="mary",
        date=datetime.datetime(year=2018, month=5, day=9, hour=16, minute=0, second=42, microsecond=0, tzinfo=None),
        request=Request(http_verb="POST", http_path=["api", "user"]),
        http_status=503,
        content_length=12)
    assert expected_log == W3CLogParser().parse(log_line)


if __name__ == '__main__':
    pytest.main()
