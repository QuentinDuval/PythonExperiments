"""
https://leetcode.com/problems/similar-string-groups/

Two strings X and Y are similar if we can swap two letters (in different positions) of X, so that it equals Y.

For example, "tars" and "rats" are similar (swapping at positions 0 and 2), and "rats" and "arts" are similar, but "star" is not similar to "tars", "rats", or "arts".

Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}.  Notice that "tars" and "arts" are in the same group even though they are not similar.  Formally, each group is such that a word is in the group if and only if it is similar to at least one other word in the group.

We are given a list A of strings.  Every string in A is an anagram of every other string in A.  How many groups are there?
"""


from collections import defaultdict
from typing import List


class Solution:
    def numSimilarGroups(self, nums: List[str]) -> int:
        """
        This is clearly a problem of connected components:
        - the key is therefore to avoid visited a node twice
        - the key is therefore to find the neighbors efficiently
        - then it is the traditional CC algorithm in bi-directed graphs

        One idea is to do a two pass algorithm:
        - one pass for the neighbors O(N ** 2) checks that can cost only O(1) - count the differing letters
        - one pass for the visitation of the graph which is in O(N ** 2) at maximum (dense graph)

        The problem is the memory occupation (and it is too slow)
        """

        '''
        n = len(nums)

        graph = {}
        for i in range(n):
            for j in range(i+1, n):
                if self.are_similar(nums[i], nums[j]):
                    graph.setdefault(i, []).append(j)
                    graph.setdefault(j, []).append(i)

        cc_count = 0
        discovered = set()
        for start in range(n):
            if start not in discovered:
                cc_count += 1
                self.visit_from(discovered, graph, start)
        return cc_count
        '''

        """
        The same approach, based on union find, it pretty nice. In such cases, it is more efficient (less memory used).
        We can also and eliminate duplicates.

        /!\ CANNOT SUMMARIZE WORDS IN COUNTS OF LETTERS (ALL EQUAL THEN)

        Complexity still is O(N ** 2).
        """

        nums = list(set(nums))
        n = len(nums)
        parents = list(range(n))
        ranks = [1] * n

        for i in range(n):
            for j in range(i + 1, n):
                if self.are_similar(nums[i], nums[j]):
                    self.union(parents, ranks, i, j)

        ccs = set()
        for i in range(n):
            ccs.add(self.find_parent(parents, i))
        return len(ccs)

    def find_parent(self, parents, i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(self, parents, ranks, i, j):
        x = self.find_parent(parents, i)
        y = self.find_parent(parents, j)
        if x == y:
            return
        if ranks[x] > ranks[y]:
            parents[y] = x
            ranks[x] += ranks[y]
        else:
            parents[x] = y
            ranks[y] += ranks[x]

    def are_similar(self, tag1, tag2):
        diff_count = 0
        for i in range(len(tag1)):
            if tag1[i] != tag2[i]:
                diff_count += 1
                if diff_count > 2:
                    return False
        return True

    '''
    def visit_from(self, discovered, graph, start):
        to_visit = [start]
        discovered.add(start)
        while to_visit:
            node = to_visit.pop()
            for neighbor in graph.get(node, []):
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    to_visit.append(neighbor)
    '''
