from functools import lru_cache


"""
https://leetcode.com/problems/valid-parenthesis-string/
Given a string containing only three types of characters: '(', ')' and '*', write a function to check whether the
parenthesis are balanced in this string (where '*' are wildcards and can be '(' or ')' or anything).
"""


def checkValidString(s: str) -> bool:
    """
    Important: the length of the string is below 100
    -------------------------------------------------------
    Backtracking is in the range of possible (branching factor is 3)
    But it requires agressive pruning:
    - Never accept a ')' if a '(' is not opened first
    - Do not accept a '(' if not enough ')' after on the right

    Is there an optimal sub-problem solution for D&C or DP?
    -------------------------------------------------------
    All the following strings are valid:
    (*), (*)), ((*), ((*))
    They show that the reason why (*) is valid does not transmit above necessarily.

    But there is still a way!
    Number of 'opened' parenthesis must be positive and is necessarily below current index.
    This means we have at most O(n ^ 2) sub-problems
    """

    @lru_cache(maxsize=None)
    def valid(i, opened) -> bool:
        if i == len(s):
            return opened == 0

        open_possible = len(s) - i - 1 >= opened
        close_possible = opened > 0

        if s[i] == '(':
            return open_possible and valid(i + 1, opened + 1)
        if s[i] == ')':
            return close_possible and valid(i + 1, opened - 1)

        return (open_possible and valid(i + 1, opened + 1)) \
               or (close_possible and valid(i + 1, opened - 1)) \
               or valid(i + 1, opened)

    return valid(i=0, opened=0)
