"""
Parser for log (could be replaced to support different formats)
"""


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

    # TODO

