"""
https://leetcode.com/problems/time-based-key-value-store/

Create a timebased key-value store class TimeMap, that supports two operations.

1. set(string key, string value, int timestamp)

    Stores the key and value, along with the given timestamp.

2. get(string key, int timestamp)

    Returns a value such that set(key, value, timestamp_prev) was called previously, with timestamp_prev <= timestamp.
    If there are multiple such values, it returns the one with the largest timestamp_prev.
    If there are no values, it returns the empty string ("").
"""


from collections import *
from typing import List


class TimeMap:
    def __init__(self):
        self.timestamps = defaultdict(list)
        self.values = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.timestamps[key].append(timestamp)
        self.values[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        timestamps = self.timestamps.get(key, None)
        if timestamps is None:
            return ""

        i = self.lower_bound(timestamps, timestamp)
        if i >= len(timestamps):
            i -= 1
        elif timestamps[i] > timestamp:
            i -= 1
        return "" if i < 0 else self.values[key][i]

    def lower_bound(self, timestamps: List[int], timestamp: int):
        lo = 0
        hi = len(timestamps) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if timestamp < timestamps[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo
