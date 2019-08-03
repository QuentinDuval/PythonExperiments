"""
https://leetcode.com/problems/numbers-with-same-consecutive-differences/

Return all non-negative integers of length N such that the absolute difference between every two consecutive digits is K.

Note that every number in the answer must not have leading zeros except for the number 0 itself.
For example, 01 has one leading zero and is invalid, but 0 is valid.

You may return the answer in any order.
"""

from typing import List


class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        """
        Beware, this is not only alternating numbers...
        - if k == 1 and n == 3, you can have "123" and not only "121"

        So you need to build the numbers by trying every alternatives.
        Beats 98%.
        """
        if n == 1:
            return list(range(10))

        def build_numbers(lead_digit):
            if k == 0:
                number = 0
                for _ in range(n):
                    number = number * 10 + lead_digit
                return [number]

            numbers = [lead_digit]
            for _ in range(n - 1):
                next_numbers = []
                for number in numbers:
                    digit = number % 10
                    if digit - k >= 0:
                        next_numbers.append(number * 10 + digit - k)
                    if digit + k <= 9:
                        next_numbers.append(number * 10 + digit + k)
                numbers = next_numbers
            return numbers

        numbers = []
        for lead_digit in range(1, 10):
            numbers.extend(build_numbers(lead_digit))
        return numbers
