"""
Given an array of citations sorted in ascending order (each citation is a non-negative integer) of a researcher,
write a function to compute the researcher's h-index.

A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each.

If there are several possible values for h, the maximum one is taken as the h-index.
Solve it in logarithmic time complexity.
"""

from typing import List


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        """
        The citations are sorted in ascending order:

        [0,1,3,5,6]
             ^

        Search for the leftmost index 'i' such that:
        - citation[i] >= len(citation) - i
        - report len(citation) - i
        - it not such index, return 0

        You can do it with binary search:
        * if citation[i] >= len(citation) - i: search on the left (look for a leftmost match)
        * if citation[i] < len(citation) - i: search on the right (look for a right match)
        """

        h_index = 0

        lo = 0
        hi = len(citations) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if citations[mid] >= len(citations) - mid:
                h_index = len(citations) - mid
                hi = mid - 1
            else:
                lo = mid + 1

        return h_index
