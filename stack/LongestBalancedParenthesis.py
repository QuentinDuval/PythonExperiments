"""
https://leetcode.com/problems/longest-valid-parentheses/
Longest substring with valid parentheses (correct number of open and closed an in the right order)
The key is SYMETRIE
"""


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


def longest_valid_parentheses(s: str) -> int:
    """
    Stack based implementation:
    - Keep the index of the opening '('
    - Pop the index when we meet a ')'
    - If nothing to pop, then retart at next position
    """
    longest = 0
    stack = [-1]  # Last valid point
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            stack.append(i)
            i += 1
        elif len(stack) == 1:
            while i < len(s) and s[i] != '(':
                i += 1
            stack[-1] = i - 1
        else:
            stack.pop()
            longest = max(longest, i - stack[-1])
            i += 1
    return longest
