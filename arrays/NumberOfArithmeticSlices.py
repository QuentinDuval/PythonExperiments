"""
https://leetcode.com/problems/arithmetic-slices/

A sequence of number is called arithmetic if it consists of at least three elements and if the difference between any
two consecutive elements is the same.

A zero-indexed array A consisting of N numbers is given.
A slice of that array is any pair of integers (P, Q) such that 0 <= P < Q < N.

A slice (P, Q) of array A is called arithmetic if the sequence:
A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular, this means that P + 1 < Q.

The function should return the number of arithmetic slices in the array A.
"""

from typing import List


class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        """
        The brute force algorithm would consist in trying all i + 2 <= j and see if it is an arithmetic sequence.

        Of course, there are optimizations to do here:
        - If the beginning of a sequence is not arithmetic, it is not useful to continue.
        - If a sequence is arithmetic in 2, it will not be arithmetic in 3 later.

        So we can identify a longest slices of length >= 3 that is arithmetic in one number,
        then advance to find the next longest slices.

        Then we need a bit of counting down possibilities:
        If a slice if of length N >= 3:
        - it has N - 2 possible slices of length 3
        - it has N - 3 possible slices of length 4
        - etc.

        =>
        Possibilities = Sum(K=3 to N) { N-K+1 }
        Possibilities = (N + 1) * (N-2) - Sum(K=3 to N) { K }
        Possibilities = (N + 1) * (N-2) - Sum(K=1 to N) { K } + 1 + 2
        Possibilities = (N + 1) * (N-2) - N * (N + 1) / 2 + 1 + 2
        """

        count = 0

        def possibilities(n):
            return (n + 1) * (n - 2) - n * (n + 1) // 2 + 1 + 2

        start = 0
        while start <= len(nums) - 3:

            i = start + 1
            while i < len(nums) and nums[i] - nums[i - 1] == nums[start + 1] - nums[start]:
                i += 1

            l = i - start
            if l >= 2:
                count += possibilities(l)
                start = i - 1  # End of sequence might be start of another sequence
            else:
                start = i  # Avoid infinite loop

        return count
