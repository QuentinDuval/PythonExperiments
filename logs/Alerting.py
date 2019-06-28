"""
Algorithm for alerting
"""

from logs.ActivityTracker import *
from logs.LogEntry import *

from typing import List


class ThroughputTracker(SlidingWindowActivityTracker):
    def __init__(self, threshold: int, window_size: int):
        super().__init__(window_size)
        self.threshold = threshold
        self.alert_start_time = None

    def transform_chunk(self, chunk: List[LogEntry]):
        return len(chunk)

    def monitor_window(self, window):
        if self.is_high_throughput(window):
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

    def is_high_throughput(self, window) -> bool:
        return sum(window) / len(window) > self.threshold

    def get_time(self):
        return repr(datetime.now())     # TODO - use the right format


class ErrorRateAlerting:
    # TODO - add some other Alert

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
