"""
https://leetcode.com/problems/insert-interval/

Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.
"""


from typing import List


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        Binary search for the first element whose 'end' is higher than
        the start of the new interval (lower_bound): O(log n)

        Binary search for the first element whose 'start' is stricly higher than
        the end of the new interval (lower_bound): O(log n)

        [[1,2],[3,5],[6,7],[8,10],[12,16]] and [4, 8]
                  ^                ^

        All elements in this open range must be fused:
        [[1,2],[3,10],[12,16]]

        Complexity stays O(N) in worst case, so we could just do a sweep on the collection...
        """
        start, end = newInterval

        # Quickly identify easy cases
        if not intervals:
            return [newInterval]
        if intervals[-1][1] < start:
            intervals.append(newInterval)
            return intervals
        if intervals[0][0] > end:
            return [newInterval] + intervals

        intervals.sort(key=lambda p: p[1])

        # Search for first interval whose end is after newInterval's start
        lpos = self.lower_bound_end(intervals, start)

        # Search for last interval whose start is before newInterval's end
        rpos = self.higher_bound_start(intervals, end)

        newInterval = min(start, intervals[lpos][0]), max(end, intervals[rpos - 1][1])
        return intervals[:lpos] + [newInterval] + intervals[rpos:]

    def lower_bound_end(self, intervals, start):
        lo = 0
        hi = len(intervals) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if intervals[mid][1] < start:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

    def higher_bound_start(self, intervals, end):
        lo = 0
        hi = len(intervals) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if intervals[mid][0] <= end:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
