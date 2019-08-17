"""
https://leetcode.com/problems/best-sightseeing-pair/

Given an array A of positive integers, A[i] represents the value of the i-th sightseeing spot, and two
sightseeing spots i and j have distance j - i between them.

The score of a pair (i < j) of sightseeing spots is (A[i] + A[j] + i - j) :
the sum of the values of the sightseeing spots, minus the distance between them.

Return the maximum score of a pair of sightseeing spots.
"""


from typing import List


class Solution:
    def maxScoreSightseeingPair_1(self, nums: List[int]) -> int:
        """
        We want to maximize A[i] + A[j] + i - j with i < j

        Group the terms and derive two arrays:
        * B[i] = A[i] - indices
        * C[i] = A[i] + indices

        Then maximize B[i] + C[i]:
        * scan from right to left, keeping the highest C[i]

        Time complexity: O(N)
        Space complexity: O(N)

        Beats 30% (596 ms)
        """

        n = len(nums)
        a = [num + i for i, num in enumerate(nums)]
        b = [num - i for i, num in enumerate(nums)]

        max_score = 0
        max_b = b[-1]
        for i in reversed(range(n - 1)):
            max_score = max(max_score, a[i] + max_b)
            max_b = max(max_b, b[i])
        return max_score

    def maxScoreSightseeingPair(self, nums: List[int]) -> int:
        """
        With slight optimization, we can get rid of tables a and b
        Time complexity is O(N)
        Space complexity down to O(1)

        Beats 42% (572 ms)
        """

        n = len(nums)
        max_score = 0
        max_b = nums[-1] - (n - 1)
        for i in reversed(range(n - 1)):
            max_score = max(max_score, nums[i] + i + max_b)
            max_b = max(max_b, nums[i] - i)
        return max_score
