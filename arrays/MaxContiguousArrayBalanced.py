"""
https://leetcode.com/problems/contiguous-array/

Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
"""


from typing import List


class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        """
        Looks a bit like a parenthesisation problem (in which you need to match "(" and ")") but the order does not count.

        We could try to keep a count with +1 for 1 and -1 for 0:
        - then we need to find the longest continuous array in which the sum is zero
        - which can be found by looking at when we first found the same value

           [0, 1, 0, 0, 1, 0]
        =>
        [0,-1, 0,-1,-2,-1,-2]
            ^           ^
        """

        longest = 0
        curr_count = 0
        first_seen = {0: -1}
        for hi, num in enumerate(nums):
            curr_count += (2 * num - 1)
            if curr_count in first_seen:
                lo = first_seen[curr_count]
                longest = max(longest, hi - lo)
            else:
                first_seen[curr_count] = hi
        return longest
