"""
https://leetcode.com/problems/number-of-ways-to-paint-n-3-grid/

You have a grid of size n x 3 and you want to paint each cell of the grid with exactly
one of the three colours: Red, Yellow or Green while making sure that no two adjacent cells
have the same colour (i.e no two cells that share vertical or horizontal sides have the same colour).

You are given n the number of rows of the grid.

Return the number of ways you can paint this grid.
As the answer may grow large, the answer must be computed modulo 10^9 + 7.
"""


class Solution:
    def numOfWays(self, n: int) -> int:
        """
        The base case for n=1 is that we have:
        - 6 types of arrangements for 2 colors (type A)
        - 6 types of arrangements for 3 colors (type B)

        We can consider these types are generic (and not worry about R,G,B):
        - type A looks like: 1 2 1
        - type B looks like: 1 2 3

        Then for induction we look at what we can do from each type:
        - from type A: 1 2 1
            - we can build 3 type A:
                2 3 2
                3 1 3
                2 1 2
            - we can build 2 type B:
                2 1 3
                3 1 2
        - from type B: 1 2 3
            - we can build 2 type A:
                2 1 2
                2 3 2
            - we can build 2 type B:
                2 3 1
                3 1 2
        Then we just recurse as many levels are needed: O(N) - beats 97%
        """

        M = 1000000007
        kind_a = 6
        kind_b = 6
        for _ in range(n - 1):
            kind_a, kind_b = (3 * kind_a + 2 * kind_b) % M, (2 * kind_a + 2 * kind_b) % M
        return (kind_a + kind_b) % M
