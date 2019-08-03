"""
https://leetcode.com/problems/longest-valid-parentheses

Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.
"""


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        """
        Does not work at all:
        Patterns such as "()()" do not obey the rule of last matching
        """

        '''
        max_valid = 0
        opened = []
        for i, c in enumerate(s):
            if c == '(':
                opened.append(i)
            elif c == ')':
                if opened:
                    matching_i = opened.pop()
                    max_valid = max(max_valid, i - matching_i + 1)
        return max_valid
        '''

        """
        Brute force would be to try all i < j and check if it is balanced: O(N ** 3) algorithm

        But we can do better:
        - scan left and count the number of parentheses open
        - keep a marker of when it last went negative
        - when it goes negative, reset this marker
        - when the opened goes to 0, diff with position of the marker
        - you have to scan from right as well
        """

        def scan(s, start_char):
            max_valid = 0
            last_reset = -1
            opened = 0
            for i, c in enumerate(s):
                if c == start_char:
                    opened += 1
                elif opened > 0:
                    opened -= 1
                    if opened == 0:
                        max_valid = max(max_valid, i - last_reset)
                else:
                    opened = 0
                    last_reset = i
            return max_valid

        return max(scan(s, '('), scan(s[::-1], ')'))
