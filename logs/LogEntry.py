"""
Data structure for the log
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=False)
class Request:
    http_verb: str
    http_path: List[str]


@dataclass(frozen=False)
class LogEntry:
    remote_host_name: str
    auth_user: str
    date: datetime
    request: Request
    http_status: int    # TODO - better type than this
    content_length: int
