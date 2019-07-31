"""
https://leetcode.com/problems/beautiful-arrangement-ii/

Given two integers n and k, you need to construct a list which contains n different positive integers ranging from 1 to n and obeys the following requirement:
Suppose this list is [a1, a2, a3, ... , an], then the list [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] has exactly k distinct integers.

If there are multiple answers, print any of them.
"""


from typing import List


class Solution:
    def constructArray(self, n: int, k: int) -> List[int]:
        """
        Following a pattern of diff:
        [max, min, max-1, min+1, max-2, min+2 ...] for the first k+1 elements

        Then just take the number in the increasing order (diff will be 1), and
        diff with the previous element will fall into the already k existing diffs
        """

        result = list(range(1, n + 1))

        i = 0
        lo = 1
        hi = k + 1
        while lo <= hi:
            result[i] = lo
            lo += 1
            i += 1
            if lo <= hi:
                result[i] = hi
                hi -= 1
                i += 1
        return result
