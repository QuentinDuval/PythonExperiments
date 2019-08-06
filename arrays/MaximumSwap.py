"""
https://leetcode.com/problems/maximum-swap/

Given a non-negative integer, you could swap two digits at most once to get the maximum valued number.
Return the maximum valued number you could get.
"""


class Solution:
    def maximumSwap(self, num: int) -> int:
        if num == 0:
            return 0

        digits = []
        while num > 0:
            quotient, remainder = divmod(num, 10)
            digits.append(remainder)
            num = quotient

        next_bigger_index = [0]
        for i in range(1, len(digits)):
            digit = digits[i]
            if digit > digits[next_bigger_index[-1]]:
                next_bigger_index.append(i)
            else:
                next_bigger_index.append(next_bigger_index[-1])

        for i in reversed(range(len(next_bigger_index))):
            next_bigger = next_bigger_index[i]
            if digits[i] != digits[next_bigger]:
                digits[i], digits[next_bigger] = digits[next_bigger], digits[i]
                break

        num = 0
        for digit in reversed(digits):
            num = num * 10 + digit
        return num
