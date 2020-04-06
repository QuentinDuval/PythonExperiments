"""
https://leetcode.com/problems/koko-eating-bananas

Koko loves to eat bananas. There are N piles of bananas, the i-th pile has piles[i] bananas.
The guards have gone and will come back in H hours.

Koko can decide her bananas-per-hour eating speed of K. Each hour, she chooses some pile of bananas, and eats K bananas
from that pile.  If the pile has less than K bananas, she eats all of them instead, and won't eat any more bananas
during this hour.

Koko likes to eat slowly, but still wants to finish eating all the bananas before the guards come back.

Return the minimum integer K such that she can eat all the bananas within H hours.
"""


import math
from typing import *


class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        """
        Binary searching for the solution:
        - first double the solution until you go below H
        - then split the solution until you find the minimum
        """

        def time_to_eat(k: int) -> int:
            time = 0
            for p in piles:
                time += math.ceil(p / k)
            return time

        lo = 1
        hi = 2
        while time_to_eat(hi) > H:
            lo, hi = hi, 2 * hi

        solution = hi
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if time_to_eat(mid) <= H:
                solution = min(solution, mid)
                hi = mid - 1
            else:
                lo = mid + 1
        return solution
