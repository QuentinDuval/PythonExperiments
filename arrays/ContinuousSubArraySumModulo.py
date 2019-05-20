"""
https://leetcode.com/problems/continuous-subarray-sum

Given a list of non-negative numbers and a target integer k, write a function to check if the array has a continuous
sub-array of size at least 2 that sums up to the multiple of k, that is, sums up to n*k where n is also an integer.

Quite interesting one...
"""


from typing import List


class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        """
        The brute force algorithm is to try all i < j + 1 and test if the sum is 0 module k
        Complexity: O(N ** 3)

        We can do better computing the sums incrementally for all i < j:
        Time Complexity: O(N ** 2)
        Space Complexity: O(1) if you do it on the run (you could get O(N) by precomputing suffix or prefix sums)
        """

        '''
        n = len(nums)
        for i in range(n):
            contiguous_sum = nums[i]
            for j in range(i+1, n):
                contiguous_sum += nums[j]
                if k == 0 and contiguous_sum == 0:
                    return True
                elif k != 0 and contiguous_sum % k == 0:
                    return True
        return False
        '''

        """
        Alternative way is to use the properties of the modulo:
        - Compute the prefix sums modulo 'k' for each position
        - If we find any place i < j + 1 where the prefix sum is equal (same number) it means we cycled => we found the sub-array
        
        Time complexity: O(N)
        Space complexity: O(N)
        """

        n = len(nums)
        k = abs(k)

        if k == 0:
            for i in range(n - 1):
                if nums[i] == 0 and nums[i + 1] == 0:
                    return True
            return False

        prefix_sums = [0]
        for n in nums:
            prefix_sums.append((prefix_sums[-1] + n) % k)

        found = set()
        for i in range(2, len(prefix_sums)):
            found.add(prefix_sums[i - 2])
            if prefix_sums[i] in found:
                return True
        return False

