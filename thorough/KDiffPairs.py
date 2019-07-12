"""
https://leetcode.com/problems/k-diff-pairs-in-an-array/

Given an array of integers and an integer k, you need to find the number of unique k-diff pairs in the array.
Here a k-diff pair is defined as an integer pair (i, j), where i and j are both numbers in the array and their absolute difference is k.
"""

from typing import List


class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        count = 0
        if k < 0:
            return 0

        if k == 0:
            histogram = {}
            for num in nums:
                histogram[num] = histogram.get(num, 0) + 1
            for v in histogram.values():
                if v > 1:
                    count += 1
        else:
            uniques = set(nums)
            for num in uniques:
                if num + k in uniques:
                    count += 1

        return count
