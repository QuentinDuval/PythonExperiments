"""
Compute useful information about a bunch of logs
"""

from collections import Counter
from logs.LogEntry import *
from typing import List


# TODO - try to use pandas here


def most_impacted_sessions(log_entries: List[LogEntry]):
    histogram = Counter(log_entry.request.http_path[0] for log_entry in log_entries if log_entry.request.http_path)
    return histogram.most_common(5)


def distribution_of_http_verbs(log_entries: List[LogEntry]):
    # TODO
    # - anomalies in error codes
    pass


def distribution_of_error_code(log_entries: List[LogEntry]):
    # TODO
    # - anomalies in error codes
    pass


def distribution_of_content_length(log_entries: List[LogEntry]):
    # TODO
    pass

