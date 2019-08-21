"""
https://leetcode.com/problems/count-of-range-sum/

Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j (i ≤ j), inclusive.
"""

from typing import List


class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        """
        Computing the prefix sum in O(n) makes it easy to compute the partial sums in O(1)
        => allows O(N**2) algo that try all i < j
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
        But requires to test all numbers in [curr_prefix_sum - upper, curr_prefix_sum - lower]

        We could use a Tree Map, but we still have to scan the numbers between
        [curr_prefix_sum - upper, curr_prefix_sum - lower] (less to test, but still)

        The solution is to use a Data Structure that allows counting in a range:
        - Binary Index Tree (with mapping of values to reduce memory footprint)
        - Segment Tree

        Complexity is O(N log N)
        Beats 45% (248 ms)
        """

        prefix_sums = [0]
        for num in nums:
            prefix_sums.append(prefix_sums[-1] + num)

        # Mapping of the different prefix_sums to indexes on the BIT
        mapping = {}
        for i, n in enumerate(sorted(set(s + d for s in prefix_sums for d in [-upper, 0, -lower]))):
            mapping[n] = i + 1

        # BIT data structure
        bit_size = len(mapping) + 2
        bit = [0] * bit_size

        def add_to_bit(val):
            val = mapping[val]
            while val < len(bit):  # Add to all ranges above by adding last bit
                bit[val] += 1
                val += val & -val

        def count_up_to(val, inclusive: bool):
            val = mapping[val] if inclusive else mapping[val] - 1
            count = 0
            while val > 0:  # Count all ranges below by removing last bit
                count += bit[val]
                val -= val & -val
            return count

        def range_count(start, end):
            return count_up_to(end, True) - count_up_to(start, False)

        total_count = 0
        for prefix_sum in prefix_sums:
            total_count += range_count(prefix_sum - upper, prefix_sum - lower)
            add_to_bit(prefix_sum)
        return total_count

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

