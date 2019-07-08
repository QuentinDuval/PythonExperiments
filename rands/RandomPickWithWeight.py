"""
https://leetcode.com/problems/random-pick-with-weight/

Given an array w of positive integers, where w[i] describes the weight of index i, write a function pickIndex which
randomly picks an index in proportion to its weight.

Note:
* 1 <= w.length <= 10000
* 1 <= w[i] <= 10^5
* pickIndex will be called at most 10000 times.
"""

import random
from typing import List


class Solution:
    def __init__(self, weights: List[int]):
        self.cum_weights = []
        total_weight = 0
        for weight in weights:
            total_weight += weight
            self.cum_weights.append(total_weight)

    def pickIndex(self) -> int:
        total_weight = self.cum_weights[-1]
        pos = random.randint(1, total_weight)   # The minimum value must be one or else first interval is bigger
        return self.lower_bound(pos)

    def lower_bound(self, val):
        lo = 0
        hi = len(self.cum_weights) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if self.cum_weights[mid] < val:     # Go right only if the middle value is strictly lower (lower bound)
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
