"""
https://leetcode.com/problems/trapping-rain-water/

Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it is able to trap after raining.
"""


from typing import List


class Solution:
    def trap(self, heights: List[int]) -> int:
        """
        The formula is simple: sum the border_height - current height for each cell.

        To compute the border_height, use DP:
        - compute the max height to the right (the border to the right) in RIGHT to LEFT pass
        - compute the max height to the left (the border to the right) during the LEFT to RIGHT pass
        """

        if not heights:
            return 0

        n = len(heights)
        right_height = [heights[-1]] * n
        for i in reversed(range(n - 1)):
            right_height[i] = max(heights[i], right_height[i + 1])

        water_trapped = 0
        left_height = heights[0]
        for i in range(1, n - 1):
            height = heights[i]
            highest_border = min(left_height, right_height[i + 1])
            if highest_border - height > 0:
                water_trapped += highest_border - height
            if height > left_height:
                left_height = height

        return water_trapped

