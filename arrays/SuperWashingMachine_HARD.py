"""
https://leetcode.com/problems/super-washing-machines/

You have n super washing machines on a line. Initially, each washing machine has some dresses or is empty.

For each move, you could choose any m (1 ≤ m ≤ n) washing machines, and pass one dress of each washing machine
to one of its adjacent washing machines at the same time .

Given an integer array representing the number of dresses in each washing machine from left to right on the line,
you should find the minimum number of moves to make all the washing machines have the same number of dresses.

If it is not possible to do it, return -1.
"""


from typing import List


class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
        """
        Do not care about individual transfers:
        * [1, 0, 5] requires 3 moves for 5 to go down to 2 (only thing to care about)
        * [0,3,0] requires 2 moves for 3 to go down to 1 (only thing to care about)
        * [1,2,2,2,3] requires 1 move (since all can move in same direction)

        But there are some subtelties:
        * [3,3,0,0,0,0] => 4 because the second number 3 will not be able to decrease at first
        * [0,3,3,0,0,0] => 3 because two sides for leaking (so the surplus of 4 will take 3 turns)
        * [0,3,0] => 2 because there are two sides, but just 1 element (max capacity of leaking)
        * [0,0,3,3,3,0,0,0,0] => 4 because of two sides

        Idea is to do a cumulative sum to check how much needs to flow:
        [100, 98, 104, 97, 105, 98, 106, 87 , 105, 98, 105, 97, 98, 101, 101]
        [0,   -2,   4, -3,   5, -2,   6, -13,   5, -2,   5, -3, -2,   1,   1]
        [0,   -2,   2, -1,   4,  2,   8,  -5,   0, -2,   3,  0, -2,  -1,   0] cum_sum_left  (V)
        [0,    0,   2, -2,   1, -4,  -2,  -8,   5,  0,   2, -3,  0,   2,   1] cum_sum_right (X)

        But you should also take into account the maximum value of each node:
        [0 ,3, 0] should be 2 cause you need 2 turns to empty one
        [-1,2,-1]
        [-1,1, 0]
        """

        # Create an example
        '''
        import numpy as np
        ex = [100] * 100
        for _ in range(500):
            i, j = np.random.randint(0,len(ex),size=2)
            ex[i] -= 1
            ex[j] += 1
        print(ex)
        '''

        # Quick check
        total = sum(machines)
        N = len(machines)
        middle = total // N
        if middle * N != total:
            return -1

        # Maximum contiguous sub-array sum
        max_surplus = max(max(m - middle, 0) for m in machines)
        cum_surplus = 0
        for hi in range(len(machines)):
            cum_surplus += (machines[hi] - middle)
            max_surplus = max(max_surplus, abs(cum_surplus))
        return max_surplus


