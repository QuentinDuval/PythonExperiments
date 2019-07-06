"""
https://leetcode.com/problems/shuffle-an-array/

Shuffle a set of numbers without duplicates.
"""

from typing import *
import random


class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums

    def reset(self) -> List[int]:
        return self.nums

    def shuffle(self) -> List[int]:
        """
        The idea is to generate a random permutation of the elements

        To generate a random permutation, the obvious way is to:
        - Pick an index from 0 to N - 1 and put it as first place
        - Pick an index from the remaining set of indices and put it as second place
        - etc

        This is completely equivalent to:
        - Pick an index from 0 to N - 1 and swap with index 0
        - Pick an index from 1 to N - 1 and swap with index 1
        - etc
        """

        n = len(self.nums)
        indexes = list(range(n))
        for i in range(n):
            j = random.randint(i, n-1)
            indexes[i], indexes[j] = indexes[j], indexes[i]
        return [self.nums[i] for i in indexes]

