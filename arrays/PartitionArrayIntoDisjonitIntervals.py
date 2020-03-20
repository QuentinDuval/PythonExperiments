"""
https://leetcode.com/problems/partition-array-into-disjoint-intervals/

Given an array A, partition it into two (contiguous) subarrays left and right so that:

* Every element in left is less than or equal to every element in right.
* left and right are non-empty.
* left has the smallest possible size.

Return the length of left after such a partitioning.
It is guaranteed that such a partitioning exists.
"""


from typing import List


class Solution:
    def partitionDisjoint(self, nums: List[int]) -> int:
        """
        Extend the 'left' window until we find an element that is higher.
        Then look at the elements on the right to see if all elements are bigger.
        To do so efficiently, just continue the scanning and keep a 'last' pointer.
        This algorithm is O(N) (two passes at max to find the max on the left)
        """
        if len(nums) < 2:
            return len(nums)

        left = 0
        max_left = nums[0]
        for i in range(1, len(nums)):
            if nums[i] >= max_left:
                continue

            for j in range(left, i + 1):
                max_left = max(max_left, nums[j])
            left = i

        return left + 1

