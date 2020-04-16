"""
https://leetcode.com/problems/reach-a-number

You are standing at position 0 on an infinite number line. There is a goal at position target.

On each move, you can either go left or right. During the n-th move (starting from 1), you take n steps.

Return the minimum number of steps required to reach the destination.
"""


import math


class Solution:
    def reachNumber(self, target: int) -> int:
        """
        The idea is to grow until we reach close or past the target using the formula N(N+1) / 2.
        Then we look by how much we overshoot:
        - if it is a pair number P, it means that we could have just taken the number P/2
          and move in the other direction
        - if it is an odd number P, it means that we could have just taken the number P/2
          and move in the other direction + add 1 (we can be done in two turns)
        """
        if target == 0:
            return 0

        # Find the first 'n' such that n (n + 1) / 2 >= target
        target = abs(target)
        n = int(math.floor(math.sqrt(2 * target)))
        while n * (n + 1) // 2 < target:
            n += 1

        # If the total matches, nothing to be done
        curr_total = n * (n + 1) // 2
        diff = curr_total - target
        if diff == 0:
            return n

        # If 'diff' is pair, it is easy, reverse the number 'diff / 2' and it is done
        if diff % 2 == 0:
            return n

        # Otherwise, if the next number if odd, we can make the diff even
        if (n+1) % 2 == 1:
            return n + 1

        # Otherwise, we have to add two numbers
        else:
            return n + 2


if __name__ == '__main__':
    solution = Solution()
    assert 0 == solution.reachNumber(0)
    assert 1 == solution.reachNumber(1)     # 1
    assert 3 == solution.reachNumber(2)     # 1, -2, 3          (diff 1 => 3 will make the diff even => +1)
    assert 2 == solution.reachNumber(3)     # 1, 2              (diff 0)
    assert 3 == solution.reachNumber(4)     # -1, 2, 3          (diff 2)
    assert 5 == solution.reachNumber(5)     # -1, 2, 3, -4, 5   (diff 1 => 4 will keep the diff odd => +2)
    assert 3 == solution.reachNumber(6)     # 1, 2, 3           (diff 0)
    assert 5 == solution.reachNumber(7)     # 1, 2, 3, -4, 5    (diff 3 => 5 will make the diff even => +1)
    assert 4 == solution.reachNumber(8)     # -1, 2, 3, 4       (diff 2)
    assert 5 == solution.reachNumber(9)     # 1, 2, -3, 4, 5    (diff 1 for 4 => 5 will make the diff even => +1)
    assert 4 == solution.reachNumber(10)    # 1, 2, 3, 4
    assert 5 == solution.reachNumber(15)
    assert 15 == solution.reachNumber(100)
    assert 47 == solution.reachNumber(1000)
