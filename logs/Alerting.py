"""
Algorithm for alerting
"""

from logs.LogEntry import *
from typing import List


# TODO - make this generic: an interface with "check_alert" that returns an Alert data structure
# TODO - the alerting process is stateful...


class ThroughputAlerting:
    def __init__(self, threshold: int, window_size: int):
        self.throughput_threshold = threshold
        self.throughput_window_size = window_size

    def is_high_throughput(self, chunk: List[LogEntry]) -> bool:
        return len(chunk) / self.throughput_window_size > self.throughput_threshold


class ErrorRateAlerting:
    def __init__(self, error_rate_threshold: int):
        self.error_rate_threshold = error_rate_threshold

    def is_high_error(self, chunk: List[LogEntry]) -> bool:
        if not chunk:
            return False

        error_count = 0
        for log_entry in chunk:
            status_category = log_entry.http_status // 100
            if status_category == 5:
                error_count += 1
        return error_count / len(chunk) > self.throughput_threshold
