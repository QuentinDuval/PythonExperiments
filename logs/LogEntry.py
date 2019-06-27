"""
Data structure for the log
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogEntry:
    remote_host_name: str
    auth_user: str
    date: str           # TODO - better type than this
    section: str
    request: str
    http_status: str    # TODO - better type than this
    content_length: int
