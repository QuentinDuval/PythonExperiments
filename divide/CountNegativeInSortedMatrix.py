"""
https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix

Given a m * n matrix grid which is sorted in non-increasing order both row-wise and column-wise.

Return the number of negative numbers in grid.
"""


from typing import List


class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0

        def bsearch(row):
            lo = 0
            hi = len(row) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if row[mid] >= 0:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return len(row) - lo

        count = 0
        h = len(grid)
        w = len(grid[0])
        for i in reversed(range(h)):
            if grid[i][-1] >= 0:
                break
            count += bsearch(grid[i])
        return count
