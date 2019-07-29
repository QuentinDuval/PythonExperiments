"""
https://leetcode.com/problems/wiggle-subsequence/

A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate
between positive and negative. The first difference (if one exists) may be either positive or negative.

A sequence with fewer than two elements is trivially a wiggle sequence.

For example, [1,7,4,9,2,5] is a wiggle sequence because the differences (6,-3,5,-7,3) are alternately positive and
negative. In contrast, [1,4,7,2,5] and [1,7,4,5,5] are not wiggle sequences, the first because its first two differences
are positive and the second because its last difference is zero.

Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence.
A subsequence is obtained by deleting some number of elements (eventually, also zero) from the original sequence,
leaving the remaining elements in their original order.
"""


from typing import List


class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        """
        When you are increasing:
        - if you find another increasing, pick this one at new high point
        - otherwise, pick the decreasing to increase the longest sub-sequence

        When you are decreasing:
        - do the exact opposite

        You have to do 2 passes: one for each starting point.

        Complexity is O(N)
        """

        if not nums:
            return 0

        def max_len(increasing: bool):
            stack = [nums[0]]
            for num in nums[1:]:
                if increasing:
                    if num > stack[-1]:
                        stack[-1] = num
                    elif num < stack[-1]:
                        stack.append(num)
                        increasing = False
                else:
                    if num < stack[-1]:
                        stack[-1] = num
                    elif num > stack[-1]:
                        stack.append(num)
                        increasing = True
            return len(stack)

        return max(max_len(True), max_len(False))
