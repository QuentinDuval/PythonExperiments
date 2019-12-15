import sys
import math
import numpy as np
from functools import lru_cache
from typing import List, NamedTuple, Tuple

"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Game
"""

Move = Tuple[int, int]

NO_MOVE = (-1, -1)
EMPTY = 0
PLAYER = 1
OPPONENT = 2


def next_player(player_id: int) -> int:
    return 3 - player_id


COMBINATIONS = [
    # diagonals
    {(0, 0), (1, 1), (2, 2)},
    {(2, 0), (1, 1), (0, 2)},
    # rows
    {(0, 0), (0, 1), (0, 2)},
    {(1, 0), (1, 1), (1, 2)},
    {(2, 0), (2, 1), (2, 2)},
    # cols
    {(0, 0), (1, 0), (2, 0)},
    {(0, 1), (1, 1), (2, 1)},
    {(0, 2), (1, 2), (2, 2)}
]


class Board:
    __slots__ = ['fields']

    def __init__(self, fields: int):
        self.fields = fields

    @classmethod
    def empty(cls):
        return cls(fields=0)

    def play(self, coord: Move, player_id: int):
        x, y = coord
        position = 2 * (x * 3 + y) + (player_id - 1)
        return Board(fields=self.fields | (1 << position))

    def is_winner(self, player_id: int) -> bool:
        for combi in COMBINATIONS:
            count = 0
            for x, y in combi:
                position = 2 * (x * 3 + y) + (player_id - 1)
                if self.fields & (1 << position):
                    count += 1
            if count == 3:
                return True
        return False

    def available_moves(self):
        for x in range(3):
            for y in range(3):
                position = 2 * (x * 3 + y)
                if not 1 << position & self.fields and not 1 << (position + 1) & self.fields:
                    yield x, y

    def __eq__(self, other):
        return self.fields == other.fields

    def __hash__(self):
        return hash(self.fields)


"""
Agent
"""


# TODO - use Dynamic Programming to compute all combinations (3 ^ 9)


class PerfectAgent:
    def __init__(self):
        pass

    # TODO - this code does not work: select a move that will win or at least not lead to certain loss

    def get_action(self, board: Board) -> Move:
        for move in board.available_moves():
            new_board = board.play(move, PLAYER)
            if self.will_win(new_board, PLAYER):
                return move

        for move in board.available_moves():
            new_board = board.play(move, PLAYER)
            if not self.will_lose(new_board, PLAYER):
                return move
        return move

    @lru_cache(maxsize=None)
    def will_lose(self, board: Board, player_id: int) -> bool:
        if board.is_winner(player_id):
            return False

        for move in board.available_moves():
            new_board = board.play(move, next_player(player_id))
            if not self.will_win(new_board, next_player(player_id)):
                return False
        return True

    @lru_cache(maxsize=None)
    def will_win(self, board: Board, player_id: int) -> bool:
        if board.is_winner(player_id):
            return True

        for move in board.available_moves():
            new_board = board.play(move, next_player(player_id))
            if not self.will_lose(new_board, next_player(player_id)):
                return False
        return True


"""
Inputs acquisition
"""


def read_coord():
    row, col = [int(i) for i in input().split()]
    return row, col


def read_valid():
    count = int(input())
    return [read_coord() for _ in range(count)]


def print_move(move):
    print(str(move[0]) + " " + str(move[1]))


"""
Game loop
"""

board = Board.empty()
agent = PerfectAgent()

while True:
    opponent_move = read_coord()
    valid_moves = read_valid()
    if opponent_move != NO_MOVE:
        board = board.play(opponent_move, OPPONENT)

    debug(valid_moves)
    debug(list(board.available_moves()))

    move = agent.get_action(board)
    board = board.play(move, PLAYER)
    print_move(move)
