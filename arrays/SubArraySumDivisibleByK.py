"""
https://leetcode.com/problems/subarray-sums-divisible-by-k/
"""

from typing import List


class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        """
        Similar to sub-array sums equal to K, but with modulo.

        The principle:
        - Move through the array, keeping a cumulative sum modulo K from the start
        - Additionally, at each step:
            - Store the cumulative sum modulo K in an hash-map
            - Look for the complement of the cumulative sum modulo K in the hash-map

        Why does it works?
        - This is simply the refinement of searching for a value in the past
        - Other techniques like binary search on indexes would be slower

        Complexity is O(N), 396 ms, beats 6%
        """

        count = 0
        prev_cum_sums = {0: 1}
        cum_sum = 0                                     # Could also be named 'prefix sum'
        for num in nums:
            cum_sum = (cum_sum + (num % k) + k) % k     # The '+ k' makes sure 'cum_sum' is positive
            count += prev_cum_sums.get(cum_sum, 0)      # Search for same modulo (valid start)
            prev_cum_sums[cum_sum] = prev_cum_sums.get(cum_sum, 0) + 1
        return count

    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        """
        Improvement is possible:
        - using an array instead of a hash-map
        - using the fact that modulo always is positive in Python
        No improvement noticed though
        """

        count = 0
        prev_cum_sums = [1] + [0] * (k - 1)
        cum_sum = 0                                     # Could also be named 'prefix sum'
        for num in nums:
            cum_sum = (cum_sum + num) % k               # % always returns a positive number
            count += prev_cum_sums[cum_sum]             # Search for same modulo (valid start)
            prev_cum_sums[cum_sum] += 1
        return count

    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        """
        Another solution is to do it in 2 passes:
        - one to group elements by cluster (same modulo K)
        - then to count the unique combinations in the cluster
        => A slightly bit faster
        """

        clusters = [0] * k
        cum_sum = 0
        for num in nums:
            cum_sum += num
            clusters[cum_sum % k] += 1

        count = clusters[0]
        for cluster in clusters:
            if cluster > 1:
                count += cluster * (cluster - 1) // 2
        return count

