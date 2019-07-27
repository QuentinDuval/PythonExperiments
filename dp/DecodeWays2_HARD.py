"""
https://leetcode.com/problems/decode-ways-ii/
"""

from functools import lru_cache


class Solution:
    def numDecodings(self, s: str) -> int:
        """
        The problem here is that we depend on the past:
        - Two '*' in success means the second '*' cannot take all values in case we group them.
        - Or other examples like this one:

        Example of *1*0
        - *0 => just 10 and 20 possible => 2 ways
        - *1 => 9 if we take '*' and '1' separately + 2 (for 11 and 21)
        => 22 ways

        The recursion therefore requires knowing about the previous character when we use it.
        The below works, but the problem is that it falls in stack overflow for big inputs.
        """

        '''
        @lru_cache(maxsize=None)
        def visit(prev: s, i: int) -> int:
            # prev gives the previous character (in case we want to group)

            if i == len(s):
                return 1 if not prev else 0

            if not prev:
                if s[i] in set("3456789"):
                    return visit("", i+1)
                elif s[i] in set("12"):
                    return visit("", i+1) + visit(s[i], i+1)
                elif s[i] == '*':
                    return 9 * visit("", i+1) + visit("1", i+1) + visit("2", i+1)
                else:   # character '0'
                    return 0
            else:
                if prev == "1":
                    if s[i] == "*":
                        return 9 * visit("", i+1)
                    else:
                        return visit("", i+1)
                else:
                    if s[i] == "*":
                        return 6 * visit("", i+1)
                    elif s[i] in set("0123456"):
                        return visit("", i+1)
                    else:
                        return 0

        return visit("", 0)
        '''

        """
        So we can turn this into a nice looking memoization table (without optimization yet)
        => Now it passes, and runs in O(N) time but also O(N) space
        => Beats only 6%
        """

        '''
        n = len(s)
        memo = { k: [0] * (n + 1) for k in ["", "1", "2"] }
        memo[""][n] = 1

        def recur(prev, i):
            if not prev:
                if s[i] in set("3456789"):
                    return memo[""][i+1]
                elif s[i] in set("12"):
                    return memo[""][i+1] + memo[s[i]][i+1]
                elif s[i] == '*':
                    return 9 * memo[""][i+1] + memo["1"][i+1] + memo["2"][i+1]
                else:   # character '0'
                    return 0
            else:
                if prev == "1":
                    if s[i] == "*":
                        return 9 * memo[""][i+1]
                    else:
                        return memo[""][i+1]
                else:
                    if s[i] == "*":
                        return 6 * memo[""][i+1]
                    elif s[i] in set("0123456"):
                        return memo[""][i+1]
                    else:
                        return 0

        for i in reversed(range(n)):
            for prev in ['', '1', '2']:
                memo[prev][i] = recur(prev, i)

        return memo[""][0] % (10 ** 9 + 7)
        '''

        """
        The next level is reached by realizing we never need more than 'i+1'
        => so we can just use a constant memory
        => But still, we only beat 6%
        """

        n = len(s)
        state = {'': 1, '1': 0, '2': 0}

        def recur(prev, i):
            if not prev:
                if s[i] in set("3456789"):
                    return state[""]
                elif s[i] in set("12"):
                    return state[""] + state[s[i]]
                elif s[i] == '*':
                    return 9 * state[""] + state["1"] + state["2"]
                else:  # character '0'
                    return 0
            else:
                if prev == "1":
                    if s[i] == "*":
                        return 9 * state[""]
                    else:
                        return state[""]
                else:
                    if s[i] == "*":
                        return 6 * state[""]
                    elif s[i] in set("0123456"):
                        return state[""]
                    else:
                        return 0

        for i in reversed(range(n)):
            new_state = {}
            for prev in ['', '1', '2']:
                new_state[prev] = recur(prev, i)
            state = new_state

        return state[""] % (10 ** 9 + 7)

