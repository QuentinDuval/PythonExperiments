"""
https://leetcode.com/problems/statistics-from-a-large-sample

We sampled integers between 0 and 255, and stored the results in an array count:
count[k] is the number of integers we sampled equal to k.

Return the minimum, maximum, mean, median, and mode of the sample respectively, as an array of floating point numbers.
The mode is guaranteed to be unique.
"""


import math
from typing import List


class Solution:
    def sampleStats(self, histogram: List[int]) -> List[float]:
        """
        The minimum and maximum are easy: look at the first and last count != 0.
        Mean is easy (a simple weighted sum would do it)
        Mode is easy (just take the index of the maximum count)
        We can binary search for the median (if we do a cumulative sum of counts).
        """

        # Basic statistics
        total = 0
        minimum = len(histogram)
        maximum = 0
        mode = 0
        mean = 0.0
        for i in range(len(histogram)):
            c = histogram[i]
            total += c
            if c != 0:
                minimum = min(i, minimum)
                maximum = i
            if histogram[i] > histogram[mode]:
                mode = i
            mean += i * c
        mean /= total

        # Binary searching for the median
        cum_sum = [histogram[0]]
        for c in histogram[1:]:
            cum_sum.append(cum_sum[-1] + c)

        def bsearch(val):
            lo = 0
            hi = len(histogram) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if cum_sum[mid] < val:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        if total & 1:
            half_lo = math.ceil(total / 2)
            half_hi = math.ceil(total / 2)
        else:
            half_lo = total // 2
            half_hi = total // 2 + 1

        median = (bsearch(half_lo) + bsearch(half_hi)) / 2
        return [minimum, maximum, mean, median, mode]


