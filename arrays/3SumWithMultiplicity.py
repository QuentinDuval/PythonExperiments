"""
https://leetcode.com/problems/3sum-with-multiplicity/

Given an integer array A, and an integer target, return the number of tuples i, j, k  such that i < j < k
and A[i] + A[j] + A[k] == target.

As the answer can be very large, return it modulo 10^9 + 7.
"""


from typing import List


class Solution:
    def threeSumMulti(self, nums: List[int], target: int) -> int:
        """
        A first idea is to sort, then compress the array:

        [1,1,2,2,3,3,4,4,5,5]
        =>
        [1,2,3,4,5]
        [2,2,2,2,2]

        Then we can try all indices i < j and find a matching completing number:
        matching = target - nums[i] - nums[j]

        The problem is that we might have several times the same element to consider.
        4 passes on the data:
        - try with different numbers (see technique above)
        - try with same low number (can be done in N)
        - try with same high number (can be done in N)
        - try to find a number that divides the whole
        """

        combinations = 0

        # Combinatorics helpers
        def combi_among(k, n):
            numerator = 1
            for i in range(k + 1, n + 1):
                numerator *= i
            denominator = 1
            for i in range(1, n - k + 1):
                denominator *= i
            return numerator // denominator

        # Compress the data
        counts = {}
        for n in nums:
            counts[n] = counts.get(n, 0) + 1
        nums = list(sorted(counts.keys()))

        # Look for combinations
        N = len(nums)
        for i in range(N):
            if nums[i] > target / 2:
                break

            # Possible triplet
            if nums[i] * 3 == target:
                count = counts.get(nums[i], 0)
                if count >= 3:
                    combinations += combi_among(3, count)

            for j in range(i + 1, N):
                partial = nums[i] + nums[j]
                if partial > target:
                    break

                # Combination of different numbers
                missing = target - partial
                if missing > nums[j]:
                    combinations += counts[nums[i]] * counts[nums[j]] * counts.get(missing, 0)

                # Two times the lowest number
                if nums[i] * 2 + nums[j] == target and counts[nums[i]] >= 2:
                    combinations += combi_among(2, counts[nums[i]]) * counts[nums[j]]

                # Two times the highest number
                if nums[i] + nums[j] * 2 == target and counts[nums[j]] >= 2:
                    combinations += combi_among(2, counts[nums[j]]) * counts[nums[i]]

        return combinations % 1000000007
