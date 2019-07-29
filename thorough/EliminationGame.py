"""
https://leetcode.com/problems/elimination-game/

There is a list of sorted integers from 1 to n. Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.

Repeat the previous step again, but this time from right to left, remove the right most number and every other number from the remaining numbers.

We keep repeating the steps again, alternating left to right and right to left, until a single number remains.

Find the last number that remains starting with a list of length n.
"""


class Solution:
    def lastRemaining(self, n: int) -> int:
        """
        Examples:

        1 2 3 4 5 6 7 8 9 10
        2 4 6 8 10
        4 8
        4

        1 2 3 4 5 6 7 8 9
        2 4 6 8
        2 6
        6

        1 2 3 4 5 6 7 8
        2 4 6 8
        2 6
        6

        We move by increase step size:
        - by size 2 at first
        - then by size -4
        - then by size 8
        - etc

        The number of times we need to recurse:
        - until n == 1
        - each time with divide n by 2 (floor division)
        => log N

        How do we find the starting point at each recursion?
        - end = prev_end + (count // 2) * diff
        - begin = end + prev_diff
        """

        start = 1
        count = n
        step = 1
        while count > 1:
            end = start + (count - 1) * step
            start = end - step * (count % 2)
            count = count // 2
            step *= -2
        return start






