"""
https://leetcode.com/problems/maximum-length-of-pair-chain

You are given n pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair (c, d) can follow another pair (a, b) if and only if b < c. Chain of pairs can be formed in this fashion.

Given a set of pairs, find the length longest chain which can be formed. You needn't use up all the given pairs. You can select pairs in any order.
"""


from typing import List


class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        """
        We could see it as a dynamic programming problem:
        - sort the pairs by their first element
        - from the end, compute the longest chain from this element
        - reuse sub-solutions of each next element

        Number of sub-solutions: O(N)
        Complexity is O(N**2) in time, O(N) in space
        """

        '''
        n = len(pairs)
        pairs.sort()

        # Eliminate items with higher end date for the same start date
        prev = None
        write = 0
        for read in range(n):
            if pairs[read][0] != prev:
                pairs[write] = pairs[read]
                prev = pairs[read][0]
                write += 1
        pairs = pairs[:write]
        n = len(pairs)

        # Dynamic programming step
        memo = [1] * n
        for i in reversed(range(n)):
            for j in reversed(range(i+1, n)):
                if pairs[i][1] < pairs[j][0]:
                    memo[i] = max(memo[i], 1 + memo[j])
                else:
                    break
        return max(memo)
        '''

        """
        But there is a much simpler solution here:
        - sort by the end of the pairs
        - scan and add to the chain if start of pair > end of pair
        """

        if not pairs:
            return 0

        count = 1
        pairs.sort(key=lambda p: p[1])  # No need for p[0] since we try them anyway
        end = pairs[0][1]
        for p in pairs[1:]:
            if p[0] > end:
                count += 1
                end = p[1]
        return count
