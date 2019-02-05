from typing import List


# https://leetcode.com/problems/trapping-rain-water


def trap_water(heights: List[int]) -> int:
    """
    Two passes to compute:
    - the highest elevated point on left
    - the highest elevated point on right

    Then for each point, just compute the difference between the minimum
    between these heights and the current height
    """

    def cumulative_heights(heights):
        cum = []
        prev = 0
        for h in heights:
            prev = max(prev, h)
            cum.append(prev)
        return cum

    l_heights = cumulative_heights(heights)
    r_heights = cumulative_heights(reversed(heights))[::-1]

    trapped = 0
    for i, h in enumerate(heights):
        trapped += min(l_heights[i], r_heights[i]) - h
    return trapped
