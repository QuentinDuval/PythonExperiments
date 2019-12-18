import numpy as np
import sys
import random
from functools import lru_cache
from typing import List, Tuple
import time


"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


MAX_TURN_TIME = 100


class Chronometer:
    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time_ns()

    def spent(self):
        current = time.time_ns()
        return self._to_ms(current - self.start_time)

    def _to_ms(self, delay):
        return delay / 1_000_000


"""
Move
"""


Move = Tuple[int, int]

NO_MOVE = (-1, -1)


def div3(x: int):
    # A specific mod 3 for small numbers
    count = 0
    while x >= 3:
        count += 1
        x -= 3
    return count, x


def decompose_move(move: Move) -> Tuple[Move, Move]:
    x1, x2 = div3(move[0])
    y1, y2 = div3(move[1])
    return (x1, y1), (x2, y2)


"""
Player
"""


PlayerId = int

EMPTY = 0
PLAYER = 1
OPPONENT = 2


def next_player(player_id: PlayerId) -> PlayerId:
    return 3 - player_id


"""
Game
"""


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
    def __init__(self, sub_boards: np.ndarray, sub_winners: np.ndarray, next_quadrant: Move):
        self.sub_boards = sub_boards
        self.sub_winners = sub_winners
        self.next_quadrant = next_quadrant

    @classmethod
    def empty(cls):
        return Board(
            sub_boards=np.zeros(shape=(3, 3), dtype=np.int32),
            sub_winners=np.zeros(shape=(3, 3), dtype=np.int32),
            next_quadrant=NO_MOVE
        )

    def is_winner(self, player_id: PlayerId) -> bool:
        for combi in COMBINATIONS:
            count = 0
            for pos in combi:
                if self.sub_winners[pos] == player_id:
                    count += 1
            if count == 3:
                return True
        return False

    def play(self, player_id: PlayerId, move: Move) -> 'new board':
        main_move, sub_move = decompose_move(move)
        sub_boards = self.sub_boards.copy()
        sub_winners = self.sub_winners.copy()
        sub_boards[main_move] = self._sub_play(sub_boards[main_move], player_id, sub_move)
        sub_winners[main_move] = player_id if self._sub_winner(sub_boards[main_move], player_id) else EMPTY
        next_quadrant = sub_move if sub_winners[sub_move] == EMPTY else NO_MOVE
        return Board(sub_boards=sub_boards, sub_winners=sub_winners, next_quadrant=next_quadrant)

    def available_moves(self) -> List[Move]:
        if self.next_quadrant != NO_MOVE:
            return self._sub_available_moves(self.next_quadrant, self.sub_boards[self.next_quadrant])
        else:
            moves = []
            for x in range(3):
                for y in range(3):
                    move = (x, y)
                    if self.sub_winners[move] == EMPTY:
                        moves.extend(self._sub_available_moves(move, self.sub_boards[move]))
            return moves

    @staticmethod
    def _sub_play(sub_board: int, player_id: PlayerId, sub_move: Move) -> int:
        x, y = sub_move
        position = 2 * (x * 3 + y) + (player_id - 1)
        return sub_board | (1 << position)

    @staticmethod
    def _sub_winner(sub_board: int, player_id: PlayerId) -> bool:
        for combi in COMBINATIONS:
            count = 0
            for x, y in combi:
                position = 2 * (x * 3 + y) + (player_id - 1)
                if sub_board & (1 << position):
                    count += 1
            if count == 3:
                return True
        return False

    @staticmethod
    def _sub_available_moves(move: Move, sub_board: int) -> List[Move]:
        shift_x, shift_y = move
        shift_x *= 3
        shift_y *= 3
        moves = []
        for x in range(3):
            for y in range(3):
                position = 2 * (x * 3 + y)
                if not 1 << position & sub_board and not 1 << (position + 1) & sub_board:
                    moves.append((shift_x + x, shift_y + y))
        return moves

    def __repr__(self):
        return repr({
            'sub_boards': self._board_repr(),
            'sub_winners': self.sub_winners,
            'next_quadrant': self.next_quadrant
        })

    def _board_repr(self):
        r = [[""] * 3 for _ in range(3)]
        for x in range(3):
            for y in range(3):
                sub_board = self.sub_boards[(x, y)]
                r[x][y] = self._sub_repr(sub_board)
        return r

    @staticmethod
    def _sub_repr(sub_board: int) -> str:
        r = ""
        for x in range(3):
            for y in range(3):
                position = 2 * (x * 3 + y)
                if 1 << position & sub_board:
                    r += "X"
                elif 1 << (position + 1) & sub_board:
                    r += "O"
                else:
                    r += "-"
            r += "\n"
        return r


"""
Agent : random agent
"""


class RandomAgent:
    def __init__(self):
        self.chooser = random.choice

    def get_action(self, board: Board) -> Move:
        moves = board.available_moves()
        return self.chooser(moves)


"""
Agent : minimax agent
"""


class MinimaxAgent:
    def __init__(self):
        self.min_score = -200
        self.max_score = 200
        # TODO - order the moves to improve the A/B pruning - how?
        # TODO - improve the evaluation function (right now it does not help in many situations) - Machine Learning?
        # TODO - use the previous minimax to direct the search (MTD methods) - BUT move change at each turn
        pass

    def get_action(self, board: Board) -> Move:
        depth = 2 if board.next_quadrant == NO_MOVE else 3
        best_score, best_move = self._minimax(board, PLAYER, alpha=self.min_score, beta=self.max_score, depth=depth)
        return best_move

    # TODO - @lru_cache(maxsize=10_000)
    def _minimax(self, board: Board, player_id: int, alpha: int, beta: int, depth: int) -> Tuple[int, Move]:
        if depth <= 0:
            # debug("eval:", self._eval_board(board))
            return self._eval_board(board), NO_MOVE

        available_moves = board.available_moves()
        if not available_moves:
            return 0, NO_MOVE

        best_move = None
        best_score = self.min_score if player_id == PLAYER else self.max_score
        for move in available_moves:
            # debug("try move:", move)
            new_board = board.play(player_id, move)
            if new_board.is_winner(player_id):
                return self._win_score(player_id), move

            score, _ = self._minimax(new_board, next_player(player_id), alpha, beta, depth=depth-1)
            if player_id == PLAYER:
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
            elif player_id == OPPONENT:
                if score < best_score:
                    best_score = score
                    best_move = move
                    beta = min(beta, score)
                    if alpha >= beta:
                        break

        return best_score, best_move

    @staticmethod
    def _eval_board(board: Board) -> int:
        return np.count_nonzero(board.sub_winners == PLAYER) - np.count_nonzero(board.sub_winners == OPPONENT)

    @staticmethod
    def _win_score(player_id: int) -> int:
        return 100 if player_id == PLAYER else -100


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


def check_available_moves(expected: List[Move], board: Board):
    computed = board.available_moves()
    expected.sort()
    computed.sort()
    if expected != computed:
        debug("WRONG MOVES!")
        debug(" * Expected:", expected)
        debug(" * But got :", computed)
        debug(board)


def game_loop():
    board = Board.empty()
    agent = MinimaxAgent()

    while True:
        opponent_move = read_coord()
        if opponent_move != NO_MOVE:
            debug("opponent move:", opponent_move)
            debug("decompose to:", decompose_move(opponent_move))
            board = board.play(OPPONENT, opponent_move)

        valid_moves = read_valid()
        check_available_moves(valid_moves, board)

        move = agent.get_action(board)
        board = board.play(PLAYER, move)
        print_move(move)


game_loop()


"""
Tests IA
"""


def test_ia():
    agent = MinimaxAgent()
    board = Board.empty()
    board = board.play(PLAYER, (0, 0))
    board = board.play(OPPONENT, (2, 2))
    board = board.play(PLAYER, (6, 6))
    board = board.play(OPPONENT, (0, 2))
    print(agent.get_action(board))


# test_ia()


"""
Test game
"""


def tests_game():
    board = Board.empty()
    available_moves = board.available_moves()
    print(available_moves)

    sub_board = 0
    sub_board = Board._sub_play(sub_board, PLAYER, (0, 0))
    sub_board = Board._sub_play(sub_board, PLAYER, (1, 1))
    sub_board = Board._sub_play(sub_board, PLAYER, (2, 2))
    assert Board._sub_winner(sub_board, PLAYER)

    sub_board = 0
    sub_board = Board._sub_play(sub_board, PLAYER, (0, 0))
    sub_board = Board._sub_play(sub_board, PLAYER, (1, 1))
    board = Board.empty()
    board.sub_boards[(0, 0)] = sub_board
    board = board.play(PLAYER, (2, 2))
    assert board.sub_winners[(0, 0)]


# tests_game()
