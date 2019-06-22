"""
https://leetcode.com/problems/longest-mountain-in-array

Let's call any (contiguous) subarray B (of A) a mountain if the following properties hold:

B.length >= 3
There exists some 0 < i < B.length - 1 such that B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
(Note that B could be any subarray of A, including the entire array A.)

Given an array A of integers, return the length of the longest mountain.

Return 0 if there is no mountain.
"""


from typing import List


class Solution:
    def longestMountain(self, heights: List[int]) -> int:
        start = None
        longest = 0
        wasDecreasing = False

        for i in range(1, len(heights)):

            if heights[i - 1] < heights[i]:
                if start is None or wasDecreasing:
                    start = i - 1
                wasDecreasing = False

            elif heights[i - 1] > heights[i]:
                if start is not None:
                    longest = max(longest, i - start + 1)
                wasDecreasing = True

            else:
                start = None
                wasDecreasing = False

        return longest
