"""
https://leetcode.com/problems/random-flip-matrix/

You are given the number of rows n_rows and number of columns n_cols of a 2D binary matrix where all values are initially 0.
Write a function flip which chooses a 0 value uniformly at random, changes it to 1, and then returns the position [row.id, col.id] of that value.
Also, write a function reset which sets all values back to 0.

Try to minimize the number of calls to system's Math.random() and optimize the time and space complexity.

Note:
- 1 <= n_rows, n_cols <= 10000
- 0 <= row.id < n_rows and 0 <= col.id < n_cols
- flip will not be called when the matrix has no 0 values left.
- the total number of calls to flip and reset will not exceed 1000.
"""

import bisect
import random
from typing import List


class Solution:
    """
    Idea 1
    ------

    Create a table with the indices inside the matrix [0 .. row * col - 1]
    To choose an index, choose among these, and everytime you choose an indice:
    - swap it at the end
    - pop it in order not to select it again

    Space complexity is O(row * col) and it blows
    """

    '''
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.indices = list(range(0, self.n_rows * self.n_cols))

    def flip(self) -> List[int]:
        i = random.randint(0, len(self.indices) - 1)
        pos = self.indices[i]
        self.indices[i], self.indices[-1] = self.indices[-1], self.indices[i]
        self.indices.pop()
        return [pos // self.n_cols, pos % self.n_cols]

    def reset(self) -> None:
        self.indices = list(range(0, self.n_rows * self.n_cols))
    '''

    """
    Idea 2
    ------

    Since we can never reuse a (row, col), we can shift the work at the beginning.
    - Shuffle 2 collections [0,rows) and [0,cols)
    - Look at each of them in succession (moving an index 'i' from 0 to rows * cols)

    Space complexity is O(row + col)
    But it is way to slow (random shuffle at 'reset' kills everything)
    """

    '''
    def __init__(self, n_rows: int, n_cols: int):
        self.rows = list(range(0, n_rows))
        self.cols = list(range(0, n_cols))
        random.shuffle(self.rows)
        random.shuffle(self.cols)
        self.i = 0

    def flip(self) -> List[int]:
        x = self.rows[self.i // len(self.cols)]
        y = self.cols[self.i % len(self.cols)]
        self.i += 1
        return [x, y]

    def reset(self) -> None:
        self.i = 0
        random.shuffle(self.rows)
        random.shuffle(self.cols)
    '''

    """
    Idea 3
    ------

    Since the number of calls to 'flip' and 'reset' is pretty small, we
    should instead of 'Idea 1' use a black list.

    To avoid calling random too many times, the goal is to pick an index
    in [0, rows *  cols - len(black_list)).

    Then we shift the index by:
    - Counting how many elements are below that index
    - And recursing as long at there is a shift
    """

    def __init__(self, n_rows: int, n_cols: int):
        self.n = n_rows * n_cols
        self.n_cols = n_cols
        self.reset()

    def flip(self) -> List[int]:
        index = random.randint(0, self.n - len(self.removed) - 1)
        alreadySkipped = 0
        while True:
            removedBefore = bisect.bisect(self.removed, index)
            if removedBefore == alreadySkipped:
                break
            index += removedBefore - alreadySkipped
            alreadySkipped = removedBefore
        bisect.insort(self.removed, index)
        return [index // self.n_cols, index % self.n_cols]

    def reset(self) -> None:
        self.removed = []
