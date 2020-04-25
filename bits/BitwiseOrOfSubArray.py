"""
https://leetcode.com/problems/bitwise-ors-of-subarrays/

We have an array A of non-negative integers.

For every (contiguous) subarray B = [A[i], A[i+1], ..., A[j]] (with i <= j), we take the bitwise OR of all the elements in B, obtaining a result A[i] | A[i+1] | ... | A[j].

Return the number of possible results.  (Results that occur more than once are only counted once in the final answer.)
"""
from typing import List


class Solution:
    def subarrayBitwiseORs(self, A: List[int]) -> int:
        """
        If we have the numbers [a, b, c, d, e, f, g]

        We want the ranges:

        [aa, bb, cc, dd, ee, ff, gg]
        [ab, bc, cd, de, ef, fg]
        [ac, bd, ce, df, eg]
        [ad, be, cf, dg]
        ...

        Because the OR does not double count the same element (ab | bc = ac), we can simply
        fuse each elements size by size, reducing the list by 1 every time.

        This is equivalent to a O(N**2) loop, not really cool... and time exceeded.
        """

        '''
        unique = set(A)
        while len(A) > 1:
            for i in range(len(A)-1):
                A[i] |= A[i+1]
                unique.add(A[i])
            A.pop()
        return len(unique)
        '''

        """
        New observations:
        - we can remove the CONSECUTIVE equal elements
        - OR is absorbing: after a while, nothing that GROW it

        So the idea is to repeating the same work:
        - at index 'i' keep the unique elements of before (starting at any index before 'i')
        - try to grow them by adding the last element: if it does not grow, it is ok
        """

        # Remove duplicates
        write = 1
        N = len(A)
        for read in range(1, N):
            if A[read] != A[write - 1]:
                A[write] = A[read]
                write += 1
        A = A[:write]

        # Growing the ranges
        unique = set()
        prev_ranges = set()
        for n in A:
            next_ranges = {r | n for r in prev_ranges}  # extend previous ranges
            next_ranges.add(n)  # start new range
            unique.update(next_ranges)
            prev_ranges = next_ranges
        return len(unique)

