import abc
from collections import deque
from typing import List

from logs.LogEntry import *


class ActivityTracker(abc.ABC):
    """
    Abstract base class for any Alerting or Analytics implementation interested in log entries
    """

    @abc.abstractmethod
    def on_log_entry_chunk(self, chunk: List[LogEntry]):
        pass


class SlidingWindowActivityTracker(ActivityTracker):
    """
    Template method pattern to abstract the pattern of tracking log entries across a rolling time window
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque()

    def on_log_entry_chunk(self, chunk: List[LogEntry]):
        self.window.append(self.transform_chunk(chunk))
        if len(self.window) == self.window_size:
            self.resize_window()
            self.monitor_window(self.window)

    def resize_window(self):
        while len(self.window) > self.window_size:
            self.window.popleft()

    @abc.abstractmethod
    def transform_chunk(self, chunk: List[LogEntry]):
        pass

    @abc.abstractmethod
    def monitor_window(self, window: deque):
        pass
