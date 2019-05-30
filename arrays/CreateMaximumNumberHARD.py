"""
https://leetcode.com/problems/create-maximum-number/

Given two arrays of length m and n with digits 0-9 representing two numbers.
Create the maximum number of length k <= m + n from digits of the two.
The relative order of the digits from the same array must be preserved.
Return an array of the k digits.
"""


from typing import List


class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        """
        The DP solution would be very complex:
        - you need to skip elements
        - we could use MaxStacks to help, but that is too complex

        The idea is to fall back on simpler sub-problems:
        (1) maximum numbers of size 'i' and 'k-i' for nums1 and nums2 (independent)
        (2) an algorithm to fuse two number to create the maximum (DP works here)

        This is useless if len(nums1) + len(nums2) == k:
        - so problem (1) can be tried only on smaller subset
        - just try to spreat the skips on nums1 and nums2
        """

        l1 = len(nums1)
        l2 = len(nums2)
        skips = l1 + l2 - k

        best_res = None
        for i in range(skips + 1):
            n1 = self.max_number(nums1, l1 - i)
            n2 = self.max_number(nums2, l2 - skips + i)
            res = self.max_merge(n1, n2)
            if best_res is None or best_res < res:
                best_res = res

        return best_res

    def max_number(self, nums, k):
        """
        IDEA:
        Iterate through 'nums' from the left:
        - keep a stack of the numbers
        - while the number number is bigger than previous pop and replace
        - unless we reached the maximum amount of skips
        """
        pass

    def max_merge(self, nums1, nums2):
        pass  # via dynamic programming
