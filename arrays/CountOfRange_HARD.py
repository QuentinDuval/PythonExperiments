"""
https://leetcode.com/problems/count-of-range-sum/

Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j (i ≤ j), inclusive.
"""


from typing import List


class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        """
        Thoughts:
        - the negative numbers means we cannot use a two finger approach (slide a window)
        - cumulative sum in O(n) to easily compute the partial sums in O(1) => allows O(N**2) algo
        - we do not have to return the ranges, only their numbers (otherwise would be stock to N**2)

        When stuck, turn to a data structure?
        """

        '''
        # Algorithm in O(N**2) that also finds the ranges
        cum_sums = [0]
        for n in nums:
            cum_sums.append(cum_sums[-1] + n)

        count = 0
        for i in range(len(cum_sums)):
            for j in range(i+1, len(cum_sums)):
                if lower <= cum_sums[j] - cum_sums[i] <= upper:
                    count += 1
        return count
        '''

        """
        In the O(N**2) algorithm above, we in fact SEARCH for previous cumulative sums
        that would, if we substract them to the current cumulative sum, fall in the range

        We can accelerate this search using a Hash Map:
        - hash map from cum_sum to count
        - try for all values in the range
        => O(n * (upper - lower + 1)) algorithm

        We could accelerate this search further using a Tree Map:
        - tree map from cum_sum to count
        - range search lower_bound(lower), upper_bound(lower)
        => O(n * (upper - lower + 1)) algorithm worst case - often faster
        """

        '''
        cum_sums = [0]
        for n in nums:
            cum_sums.append(cum_sums[-1] + n)

        count = 0
        prev_cum_sums = {}
        for cum_sum in cum_sums:
            # We want:
            #       lower <= cum_sum - prev_sum <= upper
            # =>    cum_sum - upper <= prev_sum <= cum_sum - lower
            for prev_sum in range(cum_sum - upper, cum_sum - lower + 1):
                count += prev_cum_sums.get(prev_sum, 0)
            prev_cum_sums[cum_sum] = prev_cum_sums.get(cum_sum, 0) + 1
        return count
        '''

        """
        We can make it faster by computing the cumulative sums on the run
        """

        count = 0
        cum_sum = 0
        prev_cum_sums = {0: 1}
        for n in nums:
            cum_sum += n
            # We want:
            #       lower <= cum_sum - prev_sum <= upper
            # =>    cum_sum - upper <= prev_sum <= cum_sum - lower
            for prev_sum in range(cum_sum - upper, cum_sum - lower + 1):
                count += prev_cum_sums.get(prev_sum, 0)
            prev_cum_sums[cum_sum] = prev_cum_sums.get(cum_sum, 0) + 1
        return count

        """
        We can make it faster by using a Tree Map as described above
        (C++ solution beats 94% with std::map)
        """

        '''
        int countRangeSum(vector<int>& nums, int lower, int upper) {
        long long count = 0;
        long long cum_sum = 0;
        std::map<long long, int> prev_sums = {{0, 1}};
        for (long long n: nums) {
            cum_sum += n;
            auto hi = prev_sums.upper_bound(cum_sum - lower);
            for (auto lo = prev_sums.lower_bound(cum_sum - upper); lo != hi; ++lo)
                count += lo->second;
            prev_sums[cum_sum] += 1;
        }
        return count;
        '''

        """
        Segment tree
        """

        # TODO

