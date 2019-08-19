"""
https://leetcode.com/problems/k-th-smallest-prime-fraction

A sorted list A contains 1, plus some number of primes. Then, for every p < q in the list, we consider the fraction p/q.

What is the K-th smallest fraction considered?
Return your answer as an array of ints, where answer[0] = p and answer[1] = q.

Example:
- Input: A = [1, 2, 3, 5], K = 3
- Output: [2, 5]

Explanation:
- The fractions to be considered in sorted order are: 1/5, 1/3, 2/5, 1/2, 3/5, 2/3.
- The third fraction is 2/5.
"""

from typing import List


class Solution:
    def kthSmallestPrimeFraction(self, primes: List[int], k: int) -> List[int]:
        """
        Search the smallest prime fraction based on binary search:
        - guess the result by binary searching it
        - then check that it is indeed the right result
        (same than for Kth smallest element in multiplication matrix)

        Imagine a kind of division matrix created from [1, 2, 3, 5]

        1/5 2/5 3/5  -
        1/3 2/3  -  5/3
        1/2  -  3/2 5/2
         -  2/1 3/1 5/1

        You can easily check if fraction X is the Kth by doing a binary search at each row.
        But you cannot do this, else you have to construct the matrix, and so you could just
        do a sort on the matrix and take the Kth element. Not good.

        The key is to transform the division:

            k/5 < X => k < 5X => take the floor of 5 (divisor) mutiplied by X

        But not all primes are available
        => You need then to binary search to count how many primes are below

        How do we return then the closest fraction to X?
        Look at the divisor you obtained in previous step and check them.
        """

        primes.sort()

        lo = 0                          # Lowest fraction available
        hi = primes[-1] / primes[0]     # Highest fraction available
        while lo <= hi:
            mid = lo + (hi - lo) / 2
            count, smallest_higher = self.count_lower(primes, mid)
            if count == k - 1:
                return smallest_higher
            if count < k:
                lo = mid
            else:
                hi = mid
        return None

    def count_lower(self, primes, val):
        count = 0
        smallest_higher = None
        for denom in primes:
            lo = self.lower_bound(primes, val * denom)  # Search numerator / denom > val
            count += lo if val < 1 else lo - 1          # Avoid double counting p / p
            if lo < len(primes):
                num = primes[lo]
                if not smallest_higher or smallest_higher[0] / smallest_higher[1] > num / denom:
                    smallest_higher = [num, denom]
        return count, smallest_higher

    def lower_bound(self, primes, val):
        lo = 0
        hi = len(primes) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if primes[mid] < val:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo





