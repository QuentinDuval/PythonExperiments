"""
https://leetcode.com/problems/single-element-in-a-sorted-array/

Given a sorted array consisting of only integers where every element appears exactly twice except for one element which
appears exactly once. Find this single element that appears only once.
"""
from typing import List


class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:

        """
        O(n) solution based on XOR
        """

        '''
        single = 0
        for num in nums:
            single ^= num
        return single
        '''

        """
        O(log n) solution based on binary search
        - look for an element with a neighbor equal to itself
        - if not, you found the intruder
        - if you did, check the number of elements on left and right
        - go for the odd partition
        """

        lo = 0
        hi = len(nums) - 1
        while lo <= hi:
            if lo == hi:
                return nums[lo]

            mid = lo + (hi - lo) // 2
            mid = mid - mid % 2
            if nums[mid + 1] == nums[mid]:
                lo = mid + 2
            elif nums[mid - 1] == nums[mid]:
                hi = mid - 2
            else:
                return nums[mid]
        return -1
