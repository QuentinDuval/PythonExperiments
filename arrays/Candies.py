from typing import List


"""
https://leetcode.com/problems/candy

There are N children standing in a line. Each child is assigned a rating value.

You are giving candies to these children subjected to the following requirements:
- Each child must have at least one candy.
- Children with a higher rating get more candies than their neighbors.

What is the minimum candies you must give?
"""


class Solution:
    def candy(self, ratings: List[int]) -> int:
        """
        A children with a STRICTLY higher rating must have more candies than their lesser neighbors
        No constraints on equal children

        Idea is based on two passes left to right and right to left:
        - increment by one at each increase
        - reset to 1 at each decrease or equality
        - the result is the max of the two passes

        Example:
        - [1,3,2,1]
        - Left pass [1,2,1,1]
        - Right pass [1,3,2,1]
        - Max of passes [1,3,2,1]

        Example:
        - [2,1,2,3,1,5,4,2,3,1]
        - [1,1,2,3,1,2,1,1,2,1] left
        - [2,1,1,2,1,3,2,1,2,1] right
        """
        n = len(ratings)
        if n <= 0:
            return 0

        left = [1] * n
        for i in range(1, n):
            if ratings[i - 1] < ratings[i]:
                left[i] = left[i - 1] + 1

        right = [1] * n
        for i in reversed(range(n - 1)):
            if ratings[i] > ratings[i + 1]:
                right[i] = right[i + 1] + 1

        return sum(max(left[i], right[i]) for i in range(n))

