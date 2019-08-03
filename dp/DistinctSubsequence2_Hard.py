"""
https://leetcode.com/problems/distinct-subsequences-ii/

Given a string S, count the number of distinct, non-empty subsequences of S .

Since the result may be large, return the answer modulo 10^9 + 7.
"""


class Solution:
    def distinctSubseqII(self, s: str) -> int:
        """
        Quite amazing solution based on dynamic programming

        The recurence is normally:

        P(prefix, i)
            = P(prefix + s[i], i+1)
            + P(prefix, i+1) - all cases where we select letter == s[i]

        To implement it, we have to turn it around:

        P(i) = 2 * P(i-1) - all cases where we end up with s[i]
        """

        mod = 10 ** 9 + 7
        distinct = 2

        # Last seen count at the current letter
        prev_count = collections.defaultdict(int)
        prev_count[s[0]] = 1

        for i in range(1, len(s)):
            # Twice more possibilities, but for the possibilities that already ended with s[i]
            next_distinct = (2 * distinct - prev_count[s[i]]) % mod
            prev_count[s[i]] = distinct
            distinct = next_distinct

        # Remove the solution with the empty sequence
        return (distinct - 1) % mod
