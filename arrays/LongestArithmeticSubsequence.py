"""
https://leetcode.com/problems/longest-arithmetic-sequence/

Given an array A of integers, return the length of the longest arithmetic subsequence in A.

Recall that a subsequence of A is a list A[i_1], A[i_2], ..., A[i_k] with 0 <= i_1 < i_2 < ... < i_k <= A.length - 1,
and that a sequence B is arithmetic if B[i+1] - B[i] are all the same value (for 0 <= i < B.length - 1).
"""

import bisect
from collections import defaultdict
from typing import List


class Solution:
    def longestArithSeqLength_brute(self, nums: List[int]) -> int:
        """
        Try the Brute Force solution:
        - try every starting point
        - try every diff (second point)
        - try to find the longuest sequence from that on
        => Complexity is O(N**3)
        """

        max_len = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                last = nums[j]
                diff = nums[j] - nums[i]
                # We are doing some kind of a search here:
                # If we had the map of indexes with higher numbers for each index 'i' (using a stack) => slower in practice
                # But we can have a map of the indexes with the value we look for (and skip the indices that are lower)
                for k in range(j + 1, n):
                    if nums[k] == last + diff:
                        last = nums[k]
                        count += 1
                max_len = max(max_len, count)
        return max_len

    def longestArithSeqLength(self, nums: List[int]) -> int:
        """
        Optimize the search in the inner loop:
        - Precompute the position (the indices) matching a given value
        - Using a binary search, we can quickly identify the closest index with the given value
        """

        indices = defaultdict(list)
        for i, num in enumerate(nums):
            indices[num].append(i)

        max_len = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                last = nums[j]
                diff = nums[j] - nums[i]

                k = j + 1
                while k < n:
                    nexts = indices[last + diff]
                    pos_k = bisect.bisect_left(nexts, k)
                    if pos_k < len(nexts):
                        k = nexts[pos_k] + 1
                        count += 1
                        last += diff
                    else:
                        break

                max_len = max(max_len, count)

        return max_len
