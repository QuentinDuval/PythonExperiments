"""
https://leetcode.com/problems/merge-intervals/

Given a collection of intervals, merge all overlapping intervals.
"""


from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        We cannot really do a binary search cause the end of an interval
        is not necessarily the end of the last starting...

        So O(N) is needed as we need to touch every value.

        The idea is to sort by start of interval, keep track of the max end
        of interval found, and continue incorporating intervals which start
        date is inside the max end of interval.
        """
        if not intervals:
            return []

        intervals.sort()

        merged = []
        start_range, end_range = intervals[0]

        for interval in intervals[1:]:
            if interval[0] <= end_range:
                end_range = max(end_range, interval[1])
            else:
                merged.append([start_range, end_range])
                start_range, end_range = interval

        merged.append([start_range, end_range])
        return merged
