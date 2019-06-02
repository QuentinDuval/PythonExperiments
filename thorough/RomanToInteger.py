"""
https://leetcode.com/problems/roman-to-integer
"""


class Solution:
    def romanToInt(self, s: str) -> int:
        prev = ""

        def value_of(decr, val, factor):
            if prev == decr:
                return (val - 2) * factor
            return val * factor

        count = 0
        for c in s:
            if c == 'I':
                count += 1
            elif c == 'V':
                count += value_of('I', 5, 1)
            elif c == 'X':
                count += value_of('I', 10, 1)
            elif c == 'L':
                count += value_of('X', 5, 10)
            elif c == 'C':
                count += value_of('X', 10, 10)
            elif c == 'D':
                count += value_of('C', 5, 100)
            elif c == 'M':
                count += value_of('C', 10, 100)
            prev = c
        return count
