"""
https://leetcode.com/problems/subarray-product-less-than-k/

Your are given an array of positive integers nums.

Count and print the number of (contiguous) sub arrays where the product
of all the elements in the subarray is less than k.
"""


from typing import List


class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        """
        We know that the numbers are all between 1 and 999 included.
        => Augmenting the size of the subarray augments the product (not strictly)

        We can use this property to our advantage:
        - We can compute the product starting from index 'i' equal 0.
        - Let us say we get all indices below 'j' have a product from 'i' to 'j' lesser than 'k'.
        => all indices below 'j' will also have a product up to 'j' lesser than 'k'.

        The algorithm is therefore:
        - start with i = 0
        - find the last j such that product from i to j is lesser than 'k'
        - increment i by one (divide the product)
        - extend the j such that product from i to j is less than 'k'
        - until you hit the last i
        (for each window [i to j], add the following to the count: j - i + 1)

        Simplification of this algorithm:
        - start with i = j = 0
        - if product from [i,j] is lower than k, add to count
        - else increment i enough (and divide product) so that [i,j] is lower than k
          and add to count
        - do this for every i

        Time complexity is O(N)
        Space complexity is O(1)
        """

        if not nums:
            return 0

        if k <= 1:
            return 0

        count = 0
        lo = 0
        product = 1
        for hi in range(len(nums)):
            product *= nums[hi]
            while product >= k:
                product /= nums[lo]
                lo += 1
            count += hi - lo + 1
        return count
