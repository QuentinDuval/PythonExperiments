"""
Data structure for the log
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=False)
class LogEntry:
    remote_host_name: str
    auth_user: str
    date: datetime
    section: str
    request: str
    http_status: int    # TODO - better type than this
    content_length: int
