"""
https://leetcode.com/problems/bitwise-and-of-numbers-range/

Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
"""


class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        """
        Just write down the sequence of numbers in bit forms:

            0 0 0 0
            0 0 0 1 < jump => all zeros
            0 0 1 0
            0 0 1 1
            0 1 0 0 < jump => all zeros
            0 1 0 1
            0 1 1 0
            0 1 1 1
            1 0 0 0 < jump => all zeros
            1 0 0 1
            1 0 1 0
            1 0 1 1
            1 1 0 0
            1 1 0 1
            1 1 1 0
            1 1 1 1

        Every time there is a jump from a power of 2 between M and N: everything is reset zero.
        But we need something finer grained, and for this we will reason recurisvely.
        Let us look at 12 to 15:

            1 1 0 0
            1 1 0 1
            1 1 1 0
            1 1 1 1

        We can see that the first bit is common (no jump of power of 2): so we keep it.
        We fall back to 4 to 7 by removing those first bits (removing the highest power of 2):

            1 0 0
            1 0 1
            1 1 0
            1 1 1

        Again, the first bit is common (no jump of power of 2): so we keep it.
        At the next stage, we have a jump of bits, so all lower bits are zero.

        In other words, just identify the common prefix, and remove all bits below:
        """

        count = 0
        while m != n:
            m >>= 1
            n >>= 1
            count += 1
        return m << count
