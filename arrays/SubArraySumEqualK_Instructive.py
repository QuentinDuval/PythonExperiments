import bisect
from collections import defaultdict
from typing import List


class Solution:
    def subarraySum1(self, nums: List[int], k: int) -> int:
        """
        First solution is to try all sub-array and use a bit of dynamic programming
        to reduce complexity to O(N ** 2). Timeout!
        """

        n = len(nums)
        cum_sum = [0]
        for num in nums:
            cum_sum.append(cum_sum[-1] + num)

        count = 0
        for i in range(n+1):
            for j in range(i+1, n+1):
                if cum_sum[j] - cum_sum[i] == k:
                    count += 1
        return count

    def subarraySum2(self, nums: List[int], k: int) -> int:
        """
        Key insight is that we are searching in the algorithm above:
        For each 'i', we look for all the 'j' for which the cum_sum[j] is equal to a given value
        => Try with a hash map?

        The problem is that we only want to consider cumulative sum on the right...
        => Store the indices of the cumulative sum and do a binary search on it

        => Complexity is O(N log N), 92 ms, and beats only 8%
        """

        n = len(nums)
        cum_sum = [0]
        cum_sum_index = defaultdict(list)
        for i, num in enumerate(nums):
            cum_sum.append(cum_sum[-1] + num)
            cum_sum_index[cum_sum[-1]].append(i+1)

        count = 0
        for i in range(n+1):
            searched = k + cum_sum[i]
            matching = cum_sum_index[searched]
            lo = bisect.bisect_left(matching, i+1)
            count += len(matching) - lo
        return count

    def subarraySum3(self, nums: List[int], k: int) -> int:
        """
        You can then realize that the cum_sum array is not needed
        => 80 ms and beats 14%
        """

        n = len(nums)
        cum_sum = 0
        cum_sum_index = defaultdict(list)
        for i, num in enumerate(nums):
            cum_sum += num
            cum_sum_index[cum_sum].append(i)

        count = len(cum_sum_index[k])
        cum_sum = 0
        for i in range(n):
            cum_sum += nums[i]
            matching = cum_sum_index[k + cum_sum]
            lo = bisect.bisect_left(matching, i+1)
            count += len(matching) - lo
        return count

    def subarraySum4(self, nums: List[int], k: int) -> int:
        """
        Finally, you can realize that you can do this in one-pass
        - do not look for elements in the future, but in the past
        - avoid the binary search by just storing count of matching in the past
        => O(n) complexity, 52 ms, and beats 89%
        """

        count = 0
        cum_sum = 0
        cum_sum_index = {0: 1}
        for num in nums:
            cum_sum += num
            count += cum_sum_index.get(cum_sum - k, 0)
            cum_sum_index[cum_sum] = cum_sum_index.get(cum_sum, 0) + 1
        return count
