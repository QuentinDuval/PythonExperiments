"""
Algorithm for alerting
"""

from logs.LogEntry import *
from typing import List
import enum


class ServiceStatus(enum.Enum):
    HIGH_TRAFFIC_ALERT = 1
    NORMAL_TRAFFIC = 2


def service_status(log_entries: List[LogEntry]) -> ServiceStatus:
    # TODO
    pass
