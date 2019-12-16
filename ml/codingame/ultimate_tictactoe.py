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

    def available_moves(self) -> List[Move]:
        moves = []
        for x in range(3):
            for y in range(3):
                position = 2 * (x * 3 + y)
                if not 1 << position & self.fields and not 1 << (position + 1) & self.fields:
                    moves.append((x, y))
        return moves

    def __eq__(self, other):
        return self.fields == other.fields

    def __hash__(self):
        return hash(self.fields)

    def __repr__(self):
        r = ""
        for x in range(3):
            for y in range(3):
                position = 2 * (x * 3 + y)
                if 1 << position & self.fields:
                    r += "x"
                elif 1 << (position + 1) & self.fields:
                    r += "o"
                else:
                    r += "."
            r += "\n"
        return r


"""
Agent
"""


# TODO - use Dynamic Programming to compute all combinations (3 ^ 9)


class PerfectAgent:
    def __init__(self):
        pass

    def get_action(self, board: Board) -> Move:
        best_score, best_move = self._minimax(board, PLAYER)
        debug("best move:", best_move, "with", best_score)
        return best_move

    @lru_cache(maxsize=None)
    def _minimax(self, board: Board, player_id: int) -> Tuple[int, Move]:
        available_moves = board.available_moves()
        if not available_moves:
            return 0, None

        best_move = None
        best_score = float('-inf') if player_id == PLAYER else float('inf')
        for move in available_moves:
            new_board = board.play(move, player_id)
            if new_board.is_winner(player_id):
                best_score = self._win_score(player_id)
                best_move = move
                break

            score, _ = self._minimax(new_board, next_player(player_id))
            if player_id == PLAYER:
                if score > best_score:
                    best_score = score
                    best_move = move
            elif player_id == OPPONENT:
                if score < best_score:
                    best_score = score
                    best_move = move

        # debug(board, player_id, "=>", best_score, best_move)
        return best_score, best_move

    def _win_score(self, player_id: int) -> int:
        return 1 if player_id == PLAYER else -1


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


def game_loop():
    board = Board.empty()
    agent = PerfectAgent()

    while True:
        opponent_move = read_coord()
        valid_moves = read_valid()
        if opponent_move != NO_MOVE:
            board = board.play(opponent_move, OPPONENT)

        debug(board)

        # debug(valid_moves)
        # debug(list(board.available_moves()))

        move = agent.get_action(board)
        board = board.play(move, PLAYER)
        print_move(move)


game_loop()


"""
Tests
"""


'''
def test_1():
    board = Board.empty()
    agent = PerfectAgent()

    board = board.play((0, 0), OPPONENT)
    print(board)
    board = board.play(agent.get_action(board), PLAYER)
    print(board)
    board = board.play((0, 1), OPPONENT)
    print(board)
    board = board.play(agent.get_action(board), PLAYER)
    print(board)


test_1()
'''