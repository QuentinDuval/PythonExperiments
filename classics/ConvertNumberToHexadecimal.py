"""
https://leetcode.com/problems/convert-a-number-to-hexadecimal

Given an integer, write an algorithm to convert it to hexadecimal. For negative integer, two’s complement method is used.
"""


HEXES = "0123456789abcdef"


class Solution:
    def toHex(self, num: int) -> str:
        if num == 0:
            return "0"

        if num < 0:
            num += 2 ** 32

        out = ""
        while num > 0:
            num, key = divmod(num, 16)
            out += HEXES[key]
        return out[::-1]

    def bad(self):
        '''
        negative = num < 0
        if negative:
            num *= -1

        i = 0
        bits = [0] * 32
        while num > 0:
            num, b = divmod(num, 2)
            bits[i] = b
            i += 1

        if negative:
            carry = 1
            for i in range(len(bits)):
                carry, bits[i] = divmod(1 - bits[i] + carry, 2)

        out = ""
        for i in range(0, len(bits), 4):
            val = 0
            for j in reversed(range(i, i+4)):
                val = val * 2 + bits[j]
            out += HEXES[val]

        while out[-1] == "0":
            out = out[:-1]
        return out[::-1]
        '''


