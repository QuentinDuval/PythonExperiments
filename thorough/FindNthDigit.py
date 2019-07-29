"""
https://leetcode.com/problems/nth-digit/

Find the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
"""


class Solution:
    def findNthDigit(self, n: int) -> int:
        """
        9 numbers in the range 1-9 (size 1)
        90 numbers in the range 10-99 (size 2)
        900 numbers in the range 100-999 (size 3)
        9000 numbers in the range 1000-9999 (size 4)
        90000 numbers in the range 10000-99999 (size 5)
        ...
        """

        # Find the slice in which we are
        prev = 0
        ceil = 9
        size = 1
        while n > (ceil - prev) * size:
            n -= (ceil - prev) * size
            prev, ceil = ceil, ceil * 10 + 9
            size += 1

        # Find the number and the position in that number
        q, r = divmod(n, size)
        if r == 0:
            num = prev + q
            return num % 10
        else:
            num = prev + q + 1
            digit = int(str(num)[r - 1])
            return digit
