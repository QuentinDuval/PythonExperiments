"""
Given a positive integer num, write a function which returns True if num is a perfect square else False.
"""


class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num in {0, 1}:
            return True

        lo = 2
        hi = num // 2
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            squared = mid * mid
            if squared == num:
                return True
            elif squared < num:
                lo = mid + 1
            else:
                hi = mid - 1
        return False
