"""
Parser for log (could be replaced to support different formats)
"""

from logs.LogEntry import *

# TODO: https://github.com/rory/apache-log-parser
import apache_log_parser


import datetime


class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def take_until(self, delimiter):
        end = self.text.index(delimiter, self.pos)
        token = self.text[self.pos:end]
        self.pos = end + len(delimiter)
        self.skip_spaces()
        return token

    def take_between(self, start_delimiter, end_delimiter):
        start = self.text.index(start_delimiter, self.pos)
        start = start + len(start_delimiter)
        end = self.text.index(end_delimiter, start)
        self.pos = end + len(end_delimiter)
        self.skip_spaces()
        return self.text[start:end]

    def remaining(self):
        return self.text[self.pos:]

    def skip_spaces(self):
        while self.pos < len(self.text) and self.text[self.pos] == ' ':
            self.pos += 1


class ApacheCommonLogParser:
    """
    Parser for the log format described in
    https://www.w3.org/Daemon/User/Config/Logging.html#common-logfile-format

    remotehost rfc931 authuser [date] "request" status bytes

    remotehost
        Remote hostname (or IP number if DNS hostname is not available, or if DNSLookup is Off.

    rfc931
        The remote logname of the user.

    authuser
        The username as which the user has authenticated himself.

    [date]
        Date and time of the request.

    "request"
        The request line exactly as it came from the client.

    status
        The HTTP status code returned to the client.

    bytes
        The content-length of the document transferred.

    Examples:
        127.0.0.1 - james [09/May/2018:16:00:39 +0000] "GET /report HTTP/1.0" 200 123
        127.0.0.1 - jill [09/May/2018:16:00:41 +0000] "GET /api/user HTTP/1.0" 200 234
        127.0.0.1 - frank [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 200 34
        127.0.0.1 - mary [09/May/2018:16:00:42 +0000] "POST /api/user HTTP/1.0" 503 12
    """

    def __init__(self):
        self.date_format = '%d/%B/%Y:%H:%M:%S %z'

    def parse(self, line: str) -> LogEntry:
        try:
            tokenizer = Tokenizer(line)
            host = self.parse_host(tokenizer.take_until(' '))
            tokenizer.take_until(" ")
            auth_user = self.parse_auth_user(tokenizer.take_until(' '))
            date = self.parse_date(tokenizer.take_between('[', ']'))
            request = self.parse_request(tokenizer.take_between('"', '"'))
            http_status = self.parse_status(tokenizer.take_until(' '))
            content_length = self.parse_content_length(tokenizer.remaining())
            return LogEntry(remote_host_name=host, auth_user=auth_user, date=date, request=request,
                            http_status=http_status, content_length=content_length)
        except ValueError:
            # TODO - return some errors - but the console is already taken
            return None

    @staticmethod
    def parse_host(token: str) -> str:
        return token.strip()

    @staticmethod
    def parse_auth_user(token: str) -> str:
        return token.strip()

    def parse_date(self, token: str) -> datetime:
        return datetime.datetime.strptime(token.strip(), self.date_format)

    @classmethod
    def parse_request(cls, token: str) -> Request:
        tokenizer = Tokenizer(token.strip())
        http_verb = tokenizer.take_until(' ')
        http_path = cls.parse_http_path(tokenizer.take_until(' '))
        return Request(http_verb=http_verb, http_path=http_path)

    @staticmethod
    def parse_http_path(token):
        token = token.strip('/')
        return list(token.split('/'))

    @staticmethod
    def parse_status(token: str) -> int:
        return int(token.strip())

    @staticmethod
    def parse_content_length(token: str) -> int:
        return int(token.strip())
