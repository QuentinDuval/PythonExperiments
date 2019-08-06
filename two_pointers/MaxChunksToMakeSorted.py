"""
https://leetcode.com/problems/max-chunks-to-make-sorted/

Given an array arr that is a permutation of [0, 1, ..., arr.length - 1], we split the array into some number
of "chunks" (partitions), and individually sort each chunk.

After concatenating them, the result equals the sorted array.

What is the most number of chunks we could have made?
"""


from typing import List


class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        """
        If it goes descending... then you need to reverse it in one chunk.
        If it goes increasing... then you just reverse one by one...

        This algorithm does not work, because:
        !!! IT IS NOT A REVERSE... IT IS A SORT !!!

        Instead, you can identify the place where to sort by looking at the
        maximum number found, and putting it at its correct place

        Just extend a window...
        """

        res = 0
        end_of_window = 0
        for i, num in enumerate(arr):
            end_of_window = max(end_of_window, num)
            if end_of_window == i:
                res += 1
        return res
