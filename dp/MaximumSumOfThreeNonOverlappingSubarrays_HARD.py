"""
https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/

In a given array nums of positive integers, find three non-overlapping subarrays with maximum sum.

Each subarray will be of size k, and we want to maximize the sum of all 3*k entries.

Return the result as a list of indices representing the starting position of each interval (0-indexed).
If there are multiple answers, return the lexicographically smallest one.
"""

from typing import List


class Solution:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        """
        Dynamic Programming with LEFT and RIGHT logic

        Compute the maximum sum of k-length subarray from left to right and save for each index
        left[i] = best you can do on [0:i+1]

        Compute the maximum sum of k-length subarray from right to left and save for each index
        right[i] = best you can do on [i:]

        Then do a scan from left to right of window of size k, and maximize:
        left[i-1] + sum(nums[i:i+k]) + right[i+k]

        Be very careful with the indices here.

        Complexity is O(N), 232 ms, beats 57%.
        """

        n = len(nums)
        if n < 3 * k or k == 0:
            return 0

        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)

        left = [0] * n
        left_i = [0] * n
        right = [0] * (n + 1)  # add one to right (for case of k == 1)
        right_i = [0] * (n + 1)

        for i in range(k - 1, n):
            window = prefix_sum[i + 1] - prefix_sum[i + 1 - k]
            if window > left[i - 1]:  # > cause we prefex left start
                left[i] = window
                left_i[i] = i - (k - 1)
            else:
                left[i] = left[i - 1]
                left_i[i] = left_i[i - 1]

        for i in reversed(range(n - k + 1)):
            window = prefix_sum[i + k] - prefix_sum[i]
            if window >= right[i + 1]:  # >= cause we prefex left start
                right[i] = window
                right_i[i] = i
            else:
                right[i] = right[i + 1]
                right_i[i] = right_i[i + 1]

        max_sum = 0
        a, b, c = 0, 0, 0
        for i in range(k, n - 2 * k + 1):
            curr_sum = prefix_sum[i + k] - prefix_sum[i] + left[i - 1] + right[i + k]
            if curr_sum > max_sum:
                max_sum = curr_sum
                a, b, c = left_i[i - 1], i, right_i[i + k]

        return [a, b, c]

