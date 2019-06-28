"""
Compute useful information about a bunch of logs
"""

from collections import Counter
from typing import List

from logs.ActivityTracker import *
from logs.LogEntry import *


# TODO - try to use pandas here


class Statistic(ActivityTracker):
    # TODO - rename

    def on_log_entry_chunk(self, chunk: List[LogEntry]):
        print(self.most_impacted_sessions(chunk))

    def most_impacted_sessions(self, chunk: List[LogEntry]):
        histogram = Counter(log_entry.request.http_path[0] for log_entry in chunk if log_entry.request.http_path)
        return histogram.most_common(5)

    def distribution_of_http_verbs(self, chunk: List[LogEntry]):
        # TODO
        # - anomalies in error codes
        pass

    def distribution_of_error_code(self, chunk: List[LogEntry]):
        # TODO
        # - anomalies in error codes
        pass

    def distribution_of_content_length(self, chunk: List[LogEntry]):
        # TODO
        pass

