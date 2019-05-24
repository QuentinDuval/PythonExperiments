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
        """
        if not intervals:
            return [newInterval]

        n = len(intervals)
        start, stop = newInterval

        # Quickly identify easy cases
        if intervals[-1][1] < start:
            intervals.append(newInterval)
            return intervals
        if intervals[0][0] > stop:
            return [newInterval] + intervals

        def lower():
            lo = 0
            hi = n - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                interval = intervals[mid]
                if interval[1] < start:  # Will stop at intervals[lo][1] >= start
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        def higher():
            lo = 0
            hi = n - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                interval = intervals[mid]
                if interval[0] <= stop:  # Will stop at intervals[lo][0] > stop
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        lo = lower()  # intervals[lo] has 'end' >= start
        hi = higher()  # intervals[hi] has 'start' > stop
        fused = [min(intervals[lo][0], start), max(intervals[hi - 1][1], stop)]
        return intervals[:lo] + [fused] + intervals[hi:]
