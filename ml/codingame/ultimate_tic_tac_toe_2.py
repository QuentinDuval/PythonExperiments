import itertools
import numpy as np
import sys
import random
import math
from collections import OrderedDict
from functools import lru_cache
from typing import *
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

    def is_game_over(self):
        # TODO - is game over does not even work... there might be more available moves after... limits MCTS!
        return not self.available_moves()

    def is_winner(self, player_id: PlayerId) -> bool:
        for combi in COMBINATIONS:
            count = 0
            for pos in combi:
                if self.sub_winners[pos] == player_id:
                    count += 1
            if count == 3:
                return True

        # TODO - is_winner gives wrong result...
        # TODO - np.count_nonzero(board.sub_winners == PLAYER) - np.count_nonzero(board.sub_winners == OPPONENT)

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
        r = "|"
        for x in range(3):
            for y in range(3):
                position = 2 * (x * 3 + y)
                if 1 << position & sub_board:
                    r += "X"
                elif 1 << (position + 1) & sub_board:
                    r += "O"
                else:
                    r += "-"
            r += "|"
        return r


"""
Agent : random agent
"""


class RandomAgent:
    def __init__(self):
        self.chooser = random.choice

    def opponent_action(self, move: Move):
        pass

    def get_action(self, board: Board) -> Move:
        moves = board.available_moves()
        return self.chooser(moves)


"""
Agent : minimax agent
"""


class MinimaxAgent:
    def __init__(self, player: PlayerId, max_depth: int):
        self.min_score = -200
        self.max_score = 200
        self.player = player
        self.opponent = next_player(self.player)
        self.max_depth = max_depth
        # TODO - order the moves to improve the A/B pruning - how?
        # TODO - improve the evaluation function (right now, useless in many situations) => Machine Learning?
        # TODO - use the previous minimax to direct the search (MTD methods) - BUT move change at each turn
        pass

    def opponent_action(self, move: Move):
        pass

    def get_action(self, board: Board) -> Move:
        depth = self.max_depth - 1 if board.next_quadrant == NO_MOVE else self.max_depth
        best_score, best_move = self._mini_max(board, self.player, alpha=self.min_score, beta=self.max_score, depth=depth)
        return best_move

    # TODO - @lru_cache(maxsize=10_000)
    def _mini_max(self, board: Board, player_id: int, alpha: int, beta: int, depth: int) -> Tuple[int, Move]:
        """
        Search for the best move to perform, stopping the search at a given depth to use the evaluation function
        :param player_id: the current playing player
        :param alpha:
            The best score we can achieve already (the lower bound of our search)
            => We can cut on opponent turn, if one sub-move leads to lower value than alpha
        :param beta:
            The best score we could ever achieve (the upper limit of our search) - or the best score the opponent can do
            => We can cut on our turn, if one move leads to more than this, the opponent will never allow us to go there
        """

        if depth <= 0:
            # debug("eval:", self._eval_board(board))
            return self._eval_board(board), NO_MOVE

        available_moves = board.available_moves()
        if not available_moves:
            return 0, NO_MOVE

        best_move = None
        best_score = self.min_score if player_id == self.player else self.max_score
        for move in available_moves:
            # debug("try move:", move)
            new_board = board.play(player_id, move)
            if new_board.is_winner(player_id):
                return self._win_score(player_id), move

            score, _ = self._mini_max(new_board, next_player(player_id), alpha, beta, depth=depth - 1)
            if player_id == self.player:
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    beta = min(beta, score)
                    if alpha >= beta:
                        break

        return best_score, best_move

    def _eval_board(self, board: Board) -> int:
        return np.count_nonzero(board.sub_winners == self.player) - np.count_nonzero(board.sub_winners == self.opponent)

    def _win_score(self, player_id: int) -> int:
        return 100 if player_id == self.player else -100


"""
Monte Carlo Tree Search (MCTS) agent
"""


class GameTree:
    __slots__ = ['board', 'total', 'played', 'children']

    def __init__(self, board: Board):
        self.board = board
        self.total: int = 0
        self.played: int = 0
        self.children = {move: None for move in board.available_moves()}

    def add_experience(self, score: int):
        self.total += score
        self.played += 1

    def best(self) -> Move:
        moves = []
        weights = []
        for move, node in self.children.items():
            if node is not None:
                moves.append(move)
                weights.append(110 + node.total / node.played)
        move = random.choices(moves, weights)[0]
        return move, self.children[move]

    def select(self, maximizing: bool, exploration_factor: float = 0.) -> Tuple[Move, 'child']:
        # TODO - opponent should choose the other path...
        moves = []
        weights = []
        for move, node in self.children.items():
            if node is None:
                return move, node
            moves.append(move)
            if maximizing:
                weights.append(110 + self._weight(node, exploration_factor))
            else:
                weights.append(110 - self._weight(node, exploration_factor))
        move = random.choices(moves, weights, k=1)[0]
        return move, self.children[move]

    def _weight(self, node, exploration_factor) -> float:
        return node.total / node.played + exploration_factor * math.sqrt(math.log(self.played) / node.played)

    def __repr__(self) -> str:
        return repr({
            'total': self.total,
            'played': self.played,
            'board': self.board
        })


class MCTSAgent:
    def __init__(self, player: PlayerId, exploration_factor: float):
        self.player = player
        self.opponent = next_player(self.player)
        self.exploration_factor = exploration_factor
        self.game_tree: GameTree = None

    def opponent_action(self, move: Move):
        self.game_tree = self.game_tree.children.get(move, None)

    def get_action(self, board: Board) -> Move:
        if self.game_tree is None:
            self.game_tree = GameTree(board)

        scenario_count = 0
        chrono = Chronometer()
        chrono.start()
        while chrono.spent() <= 0.9 * MAX_TURN_TIME:
            self._monte_carlo_tree_search()
            scenario_count += 1

        debug("scenarios:", scenario_count)
        debug("spent:", chrono.spent())

        move, child = self.game_tree.best()
        self.game_tree = child
        return move

    def _monte_carlo_tree_search(self):

        # Selection (of a node to be expanded)
        player_id = self.player
        node = self.game_tree
        move = None
        nodes = []
        while node is not None and not node.board.is_game_over():
            nodes.append(node)
            move, node = node.select(self.player == player_id, self.exploration_factor)
            player_id = next_player(player_id)

        # Expansion (of a node without statistics)
        if node is None:
            new_board = nodes[-1].board.play(player_id, move)
            node = GameTree(new_board)
            nodes[-1].children[move] = node
            nodes.append(node)

        # Play-out (random action until the end)
        board = node.board
        while not board.is_game_over():
            moves = board.available_moves()
            board = board.play(player_id, random.choice(moves))
            player_id = next_player(player_id)

        # Back-propagation (of the score across the tree)
        score = 100 if board.is_winner(self.player) else -100
        for node in nodes:
            node.add_experience(score)


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


def game_loop(agent):
    board = Board.empty()
    while True:
        opponent_move = read_coord()
        if opponent_move != NO_MOVE:
            # debug("opponent move:", opponent_move)
            # debug("decompose to:", decompose_move(opponent_move))
            board = board.play(OPPONENT, opponent_move)
            agent.opponent_action(opponent_move)

        valid_moves = read_valid()
        check_available_moves(valid_moves, board)

        move = agent.get_action(board)
        board = board.play(PLAYER, move)
        print_move(move)


# game_loop(agent=MinimaxAgent(player=PLAYER, max_depth=3))
# game_loop(agent=MCTSAgent(player=PLAYER, exploration_factor=1.0))


"""
Tests IA
"""


def test_ia(agent1, agent2):
    chrono = Chronometer()
    chrono.start()
    move_count = 0
    board = Board.empty()

    while not board.is_game_over():
        move = agent1.get_action(board)
        board = board.play(PLAYER, move)
        move_count += 1
        if board.available_moves():
            move = agent2.get_action(board)
            board = board.play(OPPONENT, move)
            move_count += 1
    time_spent = chrono.spent()

    print("time spent:", time_spent)
    print("move count:", move_count)
    print("time per move:", time_spent / move_count)
    print(board.is_winner(PLAYER))
    print(board.is_winner(OPPONENT))
    print(board)


# test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=2), agent2=MinimaxAgent(player=OPPONENT, max_depth=4))
test_ia(agent1=MinimaxAgent(player=PLAYER, max_depth=3), agent2=MCTSAgent(player=OPPONENT, exploration_factor=1.0))


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
