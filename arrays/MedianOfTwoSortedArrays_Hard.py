"""
https://leetcode.com/problems/median-of-two-sorted-arrays/

There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.
"""


from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        The goal is to try to find a number X (the median) such that:
        len(nums1 <= X) + len(nums2 <= X) = len(nums1 >= X) + len(nums2 >= X)
        """

        # Simplify the problem without loss of generality: len(nums1) >= len(nums2)
        if len(nums1) < len(nums2):
            nums1, nums2 = nums2, nums1

        """
        Approach based on merge sort:
        - merge the two collections until you found enough elements on the left
        - report the element you would merged right after

        Complexity is O(m + n)
        """

        nb_nums = len(nums1) + len(nums2)
        last_val = None
        i1, i2 = 0, 0

        def take_from_nums1():
            return i2 == len(nums2) or (i1 < len(nums1) and nums1[i1] < nums2[i2])

        for _ in range(nb_nums // 2):
            if take_from_nums1():
                last_val = nums1[i1]
                i1 += 1
            else:
                last_val = nums2[i2]
                i2 += 1

        next_val = nums1[i1] if take_from_nums1() else nums2[i2]
        if nb_nums % 2 == 1:
            return float(next_val)
        else:
            return (last_val + next_val) / 2

        """
        Approach based on finding the median via:
        - binary search between the lowest of nums1 and nums2, and the maximum of nums1 and nums2
        - adjust this median until we find the right split

        Complexity is O(log(max(nums) - min(nums)) * (log n + log m))

        PROBLEM: how do you even know how to adjust the median which can be a 'float'???
        """

        '''
        nb_nums = len(nums1) + len(nums2)
        half_count = nb_nums // 2

        def lower_bound(nums, val):
            lo = 0
            hi = len(nums) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if nums[mid] < val: # will stop at nums[lo] >= val
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        lo = min(nums1[0], nums2[0])
        hi = min(nums1[-1], nums2[-1])
        while lo <= hi:
            mid = lo + (hi - lo) / 2
            print(mid)
            i1 = lower_bound(nums1, mid) # index of first value >= mid in nums1
            i2 = lower_bound(nums2, mid) # index of first value >= mid in nums2
            if i1 + i2 < half_count:
                lo = mid + 1
            elif i1 + i2 > half_count:
                hi = mid - 1
            else:
                # At this point, we have i1 + i2 elements lower than mid
                # If the number of elements is odd: the lowest of these elements is the median
                # Otherwise: (lowest of these elements + the higher of the previous elements) / 2
                # !!! Beware !!! One collection might be out of bound

                next_val = None
                if i2 == len(nums2):
                    next_val = nums1[i1]
                elif i1 == len(nums1):
                    next_val = nums2[i2]
                else:
                    next_val = min(nums1[i1], nums2[i2])

                if nb_nums % 2 == 1:
                    return float(next_val)
                else:
                    prev_val = None
                    if i2 == 0:
                        prev_val = nums1[i1-1]
                    elif i2 == len(nums2):
                        prev_val = nums2[i2-1]
                    else:
                        prev_val = max(nums1[i1-1], nums2[i2-1])
                    return (next_val + prev_val) / 2
        '''

        """
        Approach based on reversing the logic:
        - Instead of looking for a median that makes both sides be equal
        - Take the indices i1 and i2 so that they sum to (m + n) // 2
        - Check the extremities (indices around i1 and i2) such that:
            nums1[i1-1] <= num2[i2]
            nums2[i2-1] <= num2[i1]
        """

        # TODO




