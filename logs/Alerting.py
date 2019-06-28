"""
Algorithm for alerting
"""

from logs.LogConsumer import *
from logs.LogEntry import *

from collections import deque
from typing import List


class ThroughputAlerting(LogConsumer):
    def __init__(self, threshold: int, window_size: int):
        self.throughput_threshold = threshold
        self.throughput_window_size = window_size
        self.window = deque()
        self.alert_start_time = None

    def on_log_entry_chunk(self, chunk: List[LogEntry]):
        self.window.append(len(chunk))
        if len(self.window) > self.throughput_window_size:
            self.window.popleft()
            if self.is_high_throughput():
                if not self.alert_start_time:
                    self.alert_start_time = self.get_time()
                self.show_alert()
            elif self.alert_start_time:
                self.show_alert_recovered()
                self.alert_start_time = None

    def show_alert(self):
        # TODO - fill the content
        print("High traffic generated an alert - hits = {value}, triggered at {time}")

    def show_alert_recovered(self):
        # TODO - fill the content
        print("High traffic alert is recovered - recovered at {time}")

    def is_high_throughput(self) -> bool:
        return sum(self.window) / len(self.window) > self.throughput_threshold

    def get_time(self):
        return repr(datetime.now())     # TODO - use the right format


class ErrorRateAlerting:
    # TODO - add some other Alter

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
        return error_count / len(chunk) >= self.error_rate_threshold
