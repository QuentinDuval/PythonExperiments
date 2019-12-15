import sys
import math
import numpy as np
from typing import List, NamedTuple, Tuple


"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Game
"""


NO_MOVE = (-1, -1)
EMPTY = 0
PLAYER = 1
OPPONENT = 2


class Board:
    def __init__(self, grid):
        self.grid = grid

    @classmethod
    def empty(cls):
        return cls(grid=np.zeros(shape=(3, 3)))

    def play(self, coord: Tuple[int, int], player_id: int):
        self.grid[coord] = player_id

    def available_moves(self):
        xs, ys = np.where(self.grid == 0)
        return list(zip(xs, ys))


"""
Inputs acquisition
"""


def read_coord():
    row, col = [int(i) for i in input().split()]
    return (row, col)


def read_valid():
    count = int(input())
    return [read_coord() for _ in range(count)]


def print_move(move):
    print(str(move[0]) + " " + str(move[1]))


"""
Game loop
"""

board = Board.empty()

while True:
    opponent_move = read_coord()
    if opponent_move != NO_MOVE:
        board.play(opponent_move, OPPONENT)

    valid_moves = read_valid()
    debug(valid_moves)
    debug(board.available_moves())

    move = valid_moves[0]
    board.play(move, PLAYER)
    print_move(move)
