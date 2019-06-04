"""
https://leetcode.com/problems/integer-to-roman
"""

class Solution:
    def intToRoman(self, num):
        roman = ""

        m, num = divmod(num, 1000)
        roman += "M" * m

        c, num = divmod(num, 100)
        roman += addDigit(c, "M", "D", "C")

        d, u = divmod(num, 10)
        roman += addDigit(d, "C", "L", "X")
        roman += addDigit(u, "X", "V", "I")

        return roman


def addDigit(digit, ten, five, unit):
    if digit == 4:
        return unit + five
    if digit == 9:
        return unit + ten
    if digit < 4:
        return unit * digit
    return five + unit * (digit - 5)
