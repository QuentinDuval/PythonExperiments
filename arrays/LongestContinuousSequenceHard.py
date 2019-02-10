from typing import List


# TODO - improve and add tests


"""
https://leetcode.com/problems/longest-consecutive-sequence

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
- Your algorithm should run in O(n) complexity.
- The array might contain duplicate elements.
"""

from typing import List


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def longestConsecutive(nums: List[int]) -> int:
    """
    The goal is to find the length, not the sequence.

    Divide and Conquer approach?
    ----------------------------
    We could try to go for a idea similar to the 'median of median' with D&C
    Partition by average value, go to the part that has the maximal length...
    But the stop condition is not clear (there might be duplicates) so does not work really...

    Any helpful containers?
    -----------------------
    - Hash Map is the only associative container with good enough complexity
    - There are other approaches with bucket-ing but that does not seem to apply here


    How can we move on with Hash Maps?
    ----------------------------------
    - We could look for the neighbors 'n-1' and 'n+1' of each value 'n' we add
    - Then we join the intervals they have
    This suggest a Union Find algorithm whose complexity matches our needs
    """
    if not nums:
        return 0

    intervals = {n: Interval(n, n) for n in nums}
    parents = {n: n for n in nums}
    sizes = {n: 1 for n in nums}

    def find(n):
        if n not in parents:
            return None
        while parents[n] != n:
            parents[n] = parents[parents[n]]
            n = parents[n]
        return n

    def union(a, b):
        a = find(a)
        b = find(b)
        interval_a = intervals[a]
        interval_b = intervals[b]
        if sizes[a] >= sizes[b]:
            interval_a.start = min(interval_a.start, interval_b.start)
            interval_a.end = max(interval_a.start, interval_b.end)
            parents[b] = a
            sizes[a] += sizes[b]
        else:
            interval_b.start = min(interval_a.start, interval_b.start)
            interval_b.end = max(interval_a.start, interval_b.end)
            parents[a] = b
            sizes[b] += sizes[a]

    for n in nums:
        left = find(n - 1)
        mid = find(n)
        right = find(n + 1)
        if left:
            union(left, mid)
        if right:
            union(mid, right)

    return max(i.end - i.start + 1 for i in intervals.values())




