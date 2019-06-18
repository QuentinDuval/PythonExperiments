"""
https://leetcode.com/problems/find-k-pairs-with-smallest-sums/

You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u,v) which consists of one element from the first array and one element from the second array.

Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums.
"""


from heapq import *


class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        """
        The smallest pair is necessarily the smallest of nums1 and nums2.

        But for the remaining pairs, this is not as easy:
        One solution is to sort by sum of pairs (put tuples (nums[i] + nums[j], i, j) in an array and sort it)
        => Complexity is O(N^2 log N) since log (N^2) is 2 log N

        We an avoid creating that much pairs if we use a heap (the usual data structure to retrieve the top K elements),
        and find a way to populate it smartly:
        - if (i, j) is selected, the next best immediate candidates are (i+1, j) and (i, j+1)
        - there might be duplicate pairs in the heap: deduplicate them

        You should see the similarity with a Shortest Path Dijsktra algorithm (except we never update the weights so
        more like a Prim's algorithm).
        """

        if not nums1 or not nums2:
            return []

        discovered = set()
        heap, pairs = [], []

        def push(i, j):
            if (i, j) not in discovered:
                heappush(heap, (nums1[i] + nums2[j], i, j))
                discovered.add((i, j))

        push(0, 0)
        while heap and len(pairs) < k:
            _, i, j = heappop(heap)
            pairs.append([nums1[i], nums2[j]])
            if j + 1 < len(nums2):
                push(i, j + 1)
            if j == 0 and i + 1 < len(nums1):
                push(i + 1, j)
        return pairs
