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


class CachedProperty(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, wrapped):
        self._attr_name = wrapped.__name__
        self._factory = wrapped

    def __get__(self, instance, owner):
        attr = self._factory(instance)              # Compute the attribute
        setattr(instance, self._attr_name, attr)    # Cache the value; hide ourselves.
        return attr


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
OPPONENT = -1
DRAW = 2


def next_player(player_id: PlayerId) -> PlayerId:
    if player_id == PLAYER:
        return OPPONENT
    else:
        return PLAYER


"""
Game
"""

SUB_COORDINATES = [
    (1, 1),                             # Middle first
    (0, 0), (2, 0), (0, 2), (2, 2),     # Corners
    (1, 0), (0, 1), (1, 2), (2, 1)      # The rest
]

ALL_COORDINATES = [
    (shift_x * 3 + x, shift_y * 3 + y)
    for x, y in SUB_COORDINATES
    for shift_x, shift_y in SUB_COORDINATES]

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
            sub_boards=np.zeros(shape=(9, 9), dtype=np.int32),
            sub_winners=np.zeros(shape=(3, 3), dtype=np.int32),
            next_quadrant=NO_MOVE
        )

    def is_game_over(self):
        return self.winner != EMPTY

    @CachedProperty
    def winner(self) -> PlayerId:
        for combi in COMBINATIONS:
            player_count = 0
            opponent_count = 0
            for pos in combi:
                if self.sub_winners[pos] == PLAYER:
                    player_count += 1
                elif self.sub_winners[pos] == OPPONENT:
                    opponent_count += 1
            if player_count == 3:
                return PLAYER
            elif opponent_count == 3:
                return OPPONENT

        player_count = 0
        opponent_count = 0
        for move in SUB_COORDINATES:
            if self.sub_winners[move] == EMPTY:
                return EMPTY
            elif self.sub_winners[move] == PLAYER:
                player_count += 1
            elif self.sub_winners[move] == OPPONENT:
                opponent_count += 1
        if player_count > opponent_count:
            return PLAYER
        elif opponent_count > player_count:
            return OPPONENT
        else:
            return DRAW

    def play(self, player_id: PlayerId, move: Move) -> 'new board':
        main_move, sub_move = decompose_move(move)
        sub_boards = self.sub_boards.copy()
        sub_winners = self.sub_winners.copy()
        sub_boards[move] = player_id
        sub_winners[main_move] = self._sub_winner(sub_boards, main_move)
        next_quadrant = NO_MOVE if sub_winners[sub_move] != EMPTY else sub_move
        return Board(sub_boards=sub_boards, sub_winners=sub_winners, next_quadrant=next_quadrant)

    @CachedProperty
    def available_moves(self) -> List[Move]:
        if self.next_quadrant != NO_MOVE:
            return self._sub_available_moves(self.sub_boards, self.next_quadrant)
        else:
            moves = []
            for quadrant in SUB_COORDINATES:
                if self.sub_winners[quadrant] == EMPTY:
                    moves.extend(self._sub_available_moves(self.sub_boards, quadrant))
            return moves

    @staticmethod
    def _sub_winner(sub_boards: np.ndarray, quadrant: Move) -> PlayerId:
        shift_x, shift_y = quadrant
        shift_x *= 3
        shift_y *= 3
        for combi in COMBINATIONS:
            player_count = 0
            opponent_count = 0
            for x, y in combi:
                position = shift_x + x, shift_y + y
                if sub_boards[position] == PLAYER:
                    player_count += 1
                elif sub_boards[position] == OPPONENT:
                    opponent_count += 1
            if player_count == 3:
                return PLAYER
            elif opponent_count == 3:
                return OPPONENT
        if Board._filled(sub_boards, quadrant):
            return DRAW
        return EMPTY

    @staticmethod
    def _filled(sub_boards: np.ndarray, quadrant: Move) -> bool:
        for position in Board._quadrant_coords(quadrant):
            if sub_boards[position] == EMPTY:
                return False
        return True

    @staticmethod
    def _sub_available_moves(sub_boards: np.ndarray, quadrant: Move) -> List[Move]:
        moves = []
        for position in Board._quadrant_coords(quadrant):
            if sub_boards[position] == EMPTY:
                moves.append(position)
        return moves

    @staticmethod
    def _quadrant_coords(quadrant: Move):
        shift_x = 3 * quadrant[0]
        shift_y = 3 * quadrant[1]
        for x, y in SUB_COORDINATES:
            yield shift_x + x, shift_y + y

    def as_board_matrix(self):
        return self.sub_boards

    def __repr__(self):
        return "Board:\n" + repr(self.sub_boards) + "\nWinners:\n" + repr(self.sub_winners) + "\nQuadrant:" + repr(self.next_quadrant)


"""
Agent : first available move agent
"""


class FirstMoveAgent:
    def __init__(self):
        pass

    def opponent_action(self, move: Move):
        pass

    def get_action(self, board: Board) -> Move:
        return board.available_moves[0]


"""
Agent : random agent
"""


class RandomAgent:
    def __init__(self):
        self.chooser = random.choice

    def opponent_action(self, move: Move):
        pass

    def get_action(self, board: Board) -> Move:
        moves = board.available_moves
        return self.chooser(moves)


"""
Agent : minimax agent
"""


class CountOwnedEvaluation:
    def __call__(self, board: Board, player_id: PlayerId) -> float:
        opponent_id = next_player(player_id)
        return np.count_nonzero(board.sub_winners == player_id) - np.count_nonzero(board.sub_winners == opponent_id)


class MinimaxAgent:
    def __init__(self, player: PlayerId, max_depth: int, eval_fct):
        self.min_score = -200
        self.max_score = 200
        self.player = player
        self.opponent = next_player(self.player)
        self.max_depth = max_depth
        self.eval_fct = eval_fct
        # TODO - order the moves to improve the A/B pruning - how?
        # TODO - improve the evaluation function (right now, useless in many situations) => Machine Learning?
        # TODO - use the previous minimax to direct the search (MTD methods) - BUT move change at each turn
        pass

    def opponent_action(self, move: Move):
        pass

    def get_action(self, board: Board) -> Move:
        # TODO - minimax at depth 1 should be possible, correct this:
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
            return self.eval_fct(board, self.player), NO_MOVE

        best_move = None
        best_score = self.min_score if player_id == self.player else self.max_score
        for move in board.available_moves:
            # debug("try move:", move)
            new_board = board.play(player_id, move)
            winner = new_board.winner
            if winner != EMPTY:
                return self._win_score(winner), move

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

    def _win_score(self, player_id: int) -> int:
        if player_id == self.player:
            return 100
        elif player_id == self.opponent:
            return -100
        return 0


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
    computed = board.available_moves
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
            # debug(board)

        valid_moves = read_valid()
        check_available_moves(valid_moves, board)

        move = agent.get_action(board)
        board = board.play(PLAYER, move)
        # debug(board)
        print_move(move)


if __name__ == '__main__':
    game_loop(agent=MinimaxAgent(player=PLAYER, max_depth=3, eval_fct=CountOwnedEvaluation()))
    # game_loop(agent=MCTSAgent(player=PLAYER, exploration_factor=1.0))
