"""
https://leetcode.com/problems/find-right-interval

Given a set of intervals, for each of the interval i, check if there exists an interval j whose start point is bigger
than or equal to the end point of the interval i, which can be called that j is on the "right" of i.

For any interval i, you need to store the minimum interval j's index, which means that the interval j
has the minimum start point to build the "right" relationship for interval i. If the interval j doesn't exist,
store -1 for the interval i. Finally, you need output the stored value of each interval as an array.

Note:
* You may assume the interval's end point is always bigger than its start point.
* You may assume none of these intervals have the same start point.
"""


from typing import List


class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        """
        Important points:
        - we need to keep the indexes (for reporting at the end)
        - we need to find the next interval with smallest start-index higher than our own

        We could sort the intervals (the problem says that no interval have the same start)
        and then do a binary search in this sorted array to find this interval.
        """

        by_start = [(x, i) for i, (x, y) in enumerate(intervals)]
        by_start.sort()

        def lower_bound(start_point: int) -> int:
            lo = 0
            hi = len(by_start) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                mid_x, mid_i = by_start[mid]
                if mid_x < start_point:
                    lo = mid + 1
                else:
                    hi = mid - 1

            if lo < len(by_start):
                return by_start[lo][1]
            return -1

        # TODO - you can also sort by end, and do a kind of 2-sum algorithm (avoid binary search)

        return [lower_bound(y) for (x, y) in intervals]
