"""
https://leetcode.com/problems/monotone-increasing-digits/

Given a non-negative integer N, find the largest number that is less than or equal to N with monotone increasing digits.

(an integer has monotone increasing digits if and only if each pair of adjacent digits x and y satisfy x <= y.)
"""
from typing import List


class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        """
        You can see the pattern here:

        1234 -> 1234
        1232 -> 1229
        1736 -> 1699
        7738 -> 6999

        Just start from the end of the digits:
        - keep track of the lowest digit to the right
        - if the current digit is lower, decrease it (it because the next lower digit)
        - if you have to decrease a digit, all next digits should be 9s
        Beats 99%
        """

        if n <= 9:
            return n

        digits = self.to_digits(n)
        lowest = digits[-1]
        for i in reversed(range(len(digits) - 1)):
            d = digits[i]
            if d <= lowest:
                lowest = d
            else:
                digits[i + 1:] = [9] * (len(digits) - 1 - i)
                digits[i] = d - 1
                lowest = d - 1
        return self.to_number(digits)

    def to_digits(self, n: int) -> List[int]:
        digits = []
        while n > 0:
            n, r = divmod(n, 10)
            digits.append(r)
        return digits[::-1]

    def to_number(self, digits: List[int]) -> int:
        n = 0
        for d in digits:
            n = n * 10 + d
        return n

