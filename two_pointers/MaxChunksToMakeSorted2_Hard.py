"""
https://leetcode.com/problems/max-chunks-to-make-sorted-ii/

Given an array arr, we split the array into some number of "chunks" (partitions), and individually sort each chunk.

After concatenating them, the result equals the sorted array.

What is the most number of chunks we could have made?
"""

from collections import defaultdict
from typing import List


class Solution:
    def maxChunksToSorted(self, nums: List[int]) -> int:
        """
        Sort the numbers
        - then advance through both lists at the same time
        - when they contain the same number of elements, add 1 and reset the list

        Example:
        [2,1,3,4,4]
        [1,2,3,4,4]
        """

        left = defaultdict(int)
        right = defaultdict(int)

        res = 0
        sorted_nums = list(sorted(nums))
        for i in range(len(nums)):
            left[nums[i]] += 1
            right[sorted_nums[i]] += 1
            if left == right:
                res += 1
                left.clear()
                right.clear()
        return res
