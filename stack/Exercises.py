import collections


"""
Evaluator
"""


# TODO


"""
Construct binary SEARCH tree (which means you have the inorder) from a pre-order traversal
"""


# TODO


"""
Max histogram column
"""


# TODO


"""
Exercise with maximum at the head???
"""


# TODO


"""
https://leetcode.com/problems/longest-valid-parentheses/
Longest substring with valid parentheses (correct number of open and closed an in the right order)
The key is SYMETRIE
"""


# TODO - stack based implementation


def longest_valid_parentheses(s: str) -> int:
    def find_longest(s, opening_char):
        longest = 0
        start = 0
        opened = 0
        for i, c in enumerate(s):
            opened = opened + 1 if c == opening_char else opened - 1
            if opened == 0:
                longest = max(longest, i - start + 1)
            elif opened < 0:
                start = i + 1
                opened = 0
        return longest
    return max(find_longest(s, '('), find_longest(s[::-1], ')'))
