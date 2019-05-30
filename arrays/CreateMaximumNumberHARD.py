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
        # we want l1 - i >= 0           => i <= l1
        # we want l2 - skips + i >= 0   => i >= skips - l2
        for i in range(max(0, skips - l2), min(l1, skips) + 1):
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
        stack = []
        for i, n in enumerate(nums):
            while stack and stack[-1] < n and len(stack) + len(nums) - i > k:
                stack.pop()
            if len(stack) < k:
                stack.append(n)
        return stack

    def max_merge(self, nums1, nums2):
        """
        Take the highest element, and in case of tie, look ahead
        """
        # TODO - pass 'best_res' in parameter to prune
        merged = []
        i1 = 0
        i2 = 0
        while i1 < len(nums1) and i2 < len(nums2):
            if nums1[i1] > nums2[i2]:
                merged.append(nums1[i1])
                i1 += 1
            elif nums1[i1] < nums2[i2]:
                merged.append(nums2[i2])
                i2 += 1
            else:
                # You cannot extend(the range [i:ii]) all... TODO: understand why
                ii1 = i1
                ii2 = i2
                while ii1 < len(nums1) and ii2 < len(nums2) and nums1[ii1] == nums2[ii2]:
                    ii1 += 1
                    ii2 += 1
                if ii1 == len(nums1):
                    merged.append(nums2[i2])
                    i2 += 1
                elif ii2 == len(nums2) or nums1[ii1] > nums2[ii2]:
                    merged.append(nums1[i1])
                    i1 += 1
                else:
                    merged.append(nums2[i2])
                    i2 += 1

        if i1 == len(nums1):
            merged.extend(nums2[i2:])
        else:
            merged.extend(nums1[i1:])
        return merged


sol = Solution()
print(sol.max_number([9, 1, 2, 8, 0], 3))
print(sol.max_number([9, 1, 2, 8, 0], 4))
print(sol.max_number([9, 1, 2, 8, 0], 1))
print(sol.max_merge([6, 5], [9, 8, 3]))
print(sol.maxNumber(nums1=[3, 4, 6, 5], nums2=[9, 1, 2, 5, 8, 3], k=5))

