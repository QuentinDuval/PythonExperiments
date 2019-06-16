from typing import List
import bisect


class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        """
        Binary search based solution:
        - add elements from right to left
        - binary search for elements higher at each element

        Complexity is O(N^2) because of the insertion in array.
        With TreeMap, would still be O(N^2) to compute the distance.

        Beats 42% (1800ms)
        """

        pair_nb = 0
        found = []
        for n in reversed(nums):
            pair_nb += bisect.bisect_left(found, n / 2)
            bisect.insort_right(found, n)
            # found.insert(bisect.bisect_right(found, n), n)
        return pair_nb


    def reversePairs(self, nums: List[int]) -> int:
        """
        Whenever there is a i < j and a range search, you should be able to use a range-search:
        - Fenwick Tree (Binary Indexed Tree)
        - Segment Tree (Range query search)

        https://www.topcoder.com/community/competitive-programming/tutorials/binary-indexed-trees/

        You cannot simply use a cumulative sum trick because as i < j, we want to go from right
        to left, adding elements in the structure and intermix it with queries:
        - query would be O(1)
        - but updates would be O(N)

        An additional problem here is the compression of the data:
        - You do not want to create a BIT of size 2^32
        - We want to compress it to [1..K], but we want to answer nums[i] > 2 * nums[j]
        - So we need a mapping from value to first index matching a value two times bigger
          (looks like upper_bound and binary search - or simply double the size of tree)

        Complexity is O(N log N)
        Beats 90% (1200ms)
        """

        def lowest(val: int) -> int:
            if val % 2 == 1:
                return val // 2
            else:
                return val // 2 - 1

        all_nums = set(nums)
        for n in nums:
            all_nums.add(lowest(n))  # To easily search for nums[i] > 2*nums[j]
        mapping = {n: i + 1 for i, n in enumerate(sorted(all_nums))}

        tree = [0] * (len(mapping) + 1)

        def add_to_tree(val: int):
            while val < len(tree):
                tree[val] += 1
                last_bit = val & -val
                val += last_bit  # add the last bit to go to next power of 2

        def query_up_to(val: int):
            count = 0
            while val:
                count += tree[val]
                last_bit = val & -val
                val -= last_bit
            return count

        pair_nb = 0
        for n in reversed(nums):
            pair_nb += query_up_to(mapping[lowest(n)])
            add_to_tree(mapping[n])
        return pair_nb
