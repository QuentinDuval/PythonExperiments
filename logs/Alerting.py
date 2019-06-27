"""
Algorithm for alerting
"""

from logs.LogEntry import *
from typing import List
import enum


class ServiceStatus(enum.Enum):
    HIGH_TRAFFIC_ALERT = 1
    NORMAL_TRAFFIC = 2


class ThroughputAlerting:
    def __init__(self, throughput_threshold: int, throughput_window_size: int):
        self.throughput_threshold = throughput_threshold
        self.throughput_window_size = throughput_window_size

    def get_status(self, chunk: List[LogEntry]) -> ServiceStatus:
        return len(chunk) / self.throughput_window_size > self.throughput_threshold
