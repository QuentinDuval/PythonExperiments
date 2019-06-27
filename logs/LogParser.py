"""
Parser for log (could be replaced to support different formats)
"""

from logs.LogEntry import *

import datetime
import re


class W3CLogParser:
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

    EXPECTED_TOKEN_COUNT = 7

    def __init__(self):
        """
        self.matcher = re.compile("([a-zA-Z1-9._-]+) (\w) (\w) [(.*)] \"(.+)\" ([1-9]{3}) ([1-9]+)")
        self.token_parsers = [
            self.parse_host,
            self.parse_log_name,
            self.parse_auth_user,
            self.parse_date,
            self.parse_request,
            self.parse_status,
            self.parse_content_length
        ]
        """
        pass

    def parse(self, line: str) -> LogEntry:
        tokens = line.split(" ")    # TODO - Does not work because of the date format...
        if len(tokens) != self.EXPECTED_TOKEN_COUNT:
            # TODO - better management of errors (return a parse result... success or error)
            return None

        return LogEntry(
            remote_host_name=self.parse_host(tokens[0]),
            auth_user=self.parse_auth_user(tokens[2]),
            date=self.parse_date(tokens[3]),
            section=self.parse_section(tokens[4]),
            request=self.parse_request(tokens[4]),
            http_status=self.parse_status(tokens[5]),
            content_length=self.parse_content_length(tokens[6])
        )

    @classmethod
    def parse_host(cls, token: str):
        return token

    @classmethod
    def parse_log_name(cls, token: str):
        return token # TODO - unused

    @classmethod
    def parse_auth_user(cls, token: str):
        return token

    @classmethod
    def parse_date(cls, token: str):
        if not token:
            # TODO - better management of errors (return a parse result... success or error)
            return None

        if token[0] != "[" or token[-1] != "]":
            return None

        "09/May/2018:16:00:39 +0000"
        date_time = datetime.datetime.strptime(token, '%d/%m/%Y:%H:%M:%S') # Missing the +0000

    @classmethod
    def parse_request(cls, token: str):
        pass

    @classmethod
    def parse_status(cls, token: str):
        pass

    @classmethod
    def parse_content_length(cls, token: str):
        pass


