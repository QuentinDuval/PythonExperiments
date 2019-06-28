import pytest

from logs.LogParser import *


def test_tokenizer():
    tokenizer = Tokenizer("abc  [fg h] \"i j\" kl")
    assert "abc" == tokenizer.take_until(" ")
    assert "fg h" == tokenizer.take_between("[", "]")
    assert "i j" == tokenizer.take_between("\"", "\"")
    assert "kl" == tokenizer.remaining()


def test_valid_log_entry_parsing():
    log_line = '127.0.0.1 - mary [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 503 12'
    expected_log = LogEntry(
        remote_host_name='127.0.0.1',
        auth_user="mary",
        date=datetime.datetime(year=2018, month=5, day=9,
                               hour=16, minute=0, second=42,
                               tzinfo=datetime.timezone.utc),
        request=Request(http_verb="POST", http_path=["api", "user"]),
        http_status=503,
        content_length=12)
    assert expected_log == ApacheCommonLogParser().parse(log_line)


def test_invalid_log_entry_parsing():
    parser = ApacheCommonLogParser()
    assert parser.parse('127.0.0.1 - mary [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 503 ab') is None
    assert parser.parse('127.0.0.1 - mary [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" xyz 12') is None
    assert parser.parse('127.0.0.1 mary [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 503 12') is None
    assert parser.parse('127.0.0.1 - mary [09/Maz/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 503 12') is None


if __name__ == '__main__':
    pytest.main()
