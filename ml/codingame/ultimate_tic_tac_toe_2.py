import abc
import math
import random
import sys
import time
from typing import *

import numpy as np


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
    if x >= 6:
        return 2, x - 6
    if x >= 3:
        return 1, x - 3
    return 0, x


def decompose_move(move: Move) -> Tuple[Move, Move]:
    x1, x2 = div3(move[0])
    y1, y2 = div3(move[1])
    return (x1, y1), (x2, y2)


"""
Player
"""

Reward = float
PlayerId = int

EMPTY = 0
CROSS = 1
CIRCLE = -1
DRAW = 2


def next_player(player_id: PlayerId) -> PlayerId:
    if player_id == CROSS:
        return CIRCLE
    else:
        return CROSS


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
    [(0, 0), (1, 1), (2, 2)],
    [(2, 0), (1, 1), (0, 2)],
    # rows
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    # cols
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)]
]


class Board:
    __slots__ = ['grid', 'sub_winners', 'next_quadrant', 'winner', 'available_moves']

    def __init__(self,
                 grid: np.ndarray,
                 sub_winners: np.ndarray,
                 next_quadrant: Move,
                 winner: PlayerId,
                 available_moves: List[Move]):
        self.grid = grid
        self.sub_winners = sub_winners
        self.next_quadrant = next_quadrant
        self.winner = winner
        self.available_moves = available_moves

    def as_board_matrix(self):
        # TODO - encode the available_moves inside this matrix
        return self.grid

    @classmethod
    def empty(cls):
        return Board(
            grid=np.zeros(shape=(9, 9), dtype=np.int32),
            sub_winners=np.zeros(shape=(3, 3), dtype=np.int32),
            next_quadrant=NO_MOVE,
            winner=EMPTY,
            available_moves=ALL_COORDINATES
        )

    def clone(self):
        return Board(
            grid=self.grid.copy(),
            sub_winners=self.sub_winners.copy(),
            next_quadrant=self.next_quadrant,
            winner=self.winner,
            available_moves=self.available_moves
        )

    def is_over(self):
        return self.winner != EMPTY

    def play_debug(self, player_id: PlayerId, move: Move):
        if move not in self.available_moves:
            raise Exception("Invalid move", move, "in grid", self)
        return self.play(player_id, move)

    def play(self, player_id: PlayerId, move: Move) -> 'new board':
        next_board = self.clone()
        next_board.play_(player_id, move)
        return next_board

    def play_(self, player_id: PlayerId, move: Move):
        main_move, sub_move = decompose_move(move)
        self.grid[move] = player_id
        self.sub_winners[main_move] = self._sub_winner(main_move)
        self.next_quadrant = NO_MOVE if self.sub_winners[sub_move] != EMPTY else sub_move
        self.winner = self._winner()
        self.available_moves = self._available_moves()

    def _winner(self) -> PlayerId:
        # +1 for CROSS, -1 for CIRCLE, if total is 3 or -3, you have a winner
        for combi in COMBINATIONS:
            count = 0
            for pos in combi:
                count += self.sub_winners[pos]
            if count == 3:
                return CROSS
            elif count == -3:
                return CIRCLE

        # +1 for CROSS, -1 for CIRCLE, the sign gives you the winner
        count = 0
        for move in SUB_COORDINATES:
            if self.sub_winners[move] == EMPTY:
                return EMPTY
            count += self.sub_winners[move]
        if count > 0:
            return CROSS
        elif count < 0:
            return CIRCLE
        return DRAW

    def _available_moves(self) -> List[Move]:
        if self.next_quadrant != NO_MOVE:
            return self._sub_available_moves(self.next_quadrant)
        else:
            moves = []
            for quadrant in SUB_COORDINATES:
                if self.sub_winners[quadrant] == EMPTY:
                    moves.extend(self._sub_available_moves(quadrant))
            return moves

    def _sub_winner(self, quadrant: Move) -> PlayerId:
        sub_board = self._quadrant(quadrant)
        for combi in COMBINATIONS:
            count = 0   # +1 for CROSS, -1 for CIRCLE, if total is 3 or -3, you have a winner
            for pos in combi:
                count += sub_board[pos]
            if count == 3:
                return CROSS
            elif count == -3:
                return CIRCLE
        if np.count_nonzero(sub_board) == 9:
            return DRAW
        return EMPTY

    def _sub_available_moves(self, quadrant: Move) -> List[Move]:
        return [position for position in Board._quadrant_coords(quadrant) if self.grid[position] == EMPTY]

    def _quadrant(self, quadrant: Move):
        shift_x, shift_y = quadrant
        return self.grid[shift_x * 3:(shift_x + 1) * 3, shift_y * 3:(shift_y + 1) * 3]

    @staticmethod
    def _quadrant_coords(quadrant: Move):
        for x, y in SUB_COORDINATES:
            yield 3 * quadrant[0] + x, 3 * quadrant[1] + y

    def __repr__(self):
        return "Board:\n" + repr(self.grid)\
               + "\nWinners:\n" + repr(self.sub_winners)\
               + "\nQuadrant: " + repr(self.next_quadrant)\
               + "\nMoves: " + repr(self.available_moves)


"""
Definition of an agent
"""


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, board: Board, player_id: PlayerId) -> Move:
        pass

    def on_end_episode(self):
        """
        Allows stateful agents (cleaning memory) vs Reflex agents (acting on percepts only)
        """
        pass

    def on_opponent_action(self, move: Move, player_id: PlayerId):
        """
        Allows stateful agents (cleaning memory) vs Reflex agents (acting on percepts only)
        """
        pass


"""
Basic agents (random agent, first move agent, etc)
"""


class FirstMoveAgent(Agent):
    def get_action(self, board: Board, player_id: PlayerId) -> Move:
        return board.available_moves[0]


class LastMoveAgent(Agent):
    def get_action(self, board: Board, player_id: PlayerId) -> Move:
        return board.available_moves[-1]


class RandomAgent(Agent):
    def get_action(self, board: Board, player_id: PlayerId) -> Move:
        moves = board.available_moves
        return random.choice(moves)


"""
Agent : minimax agent
"""


class EvaluationFct(abc.ABC):
    @abc.abstractmethod
    def __call__(self, board: Board, player_id: PlayerId) -> float:
        pass


class MinimaxAgent(Agent):
    def __init__(self, max_depth: int, eval_fct: EvaluationFct):
        self.min_score = -200
        self.max_score = 200
        self.max_depth = max_depth
        self.eval_fct = eval_fct
        # TODO - order the moves to improve the A/B pruning - how?
        # TODO - use the previous minimax to direct the search (MTD methods) - BUT move change at each turn

    def get_action(self, board: Board, player_id: PlayerId) -> Move:
        depth = max(1, self.max_depth - 1 if board.next_quadrant == NO_MOVE else self.max_depth)
        best_score, best_move = self._mini_max(board, player_id, player_id,
                                               alpha=self.min_score, beta=self.max_score, depth=depth)
        return best_move

    # TODO - @lru_cache(maxsize=10_000)
    def _mini_max(self, board: Board, player_id: int, current_player_id: int, alpha: int, beta: int, depth: int) -> Tuple[float, Move]:
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
            return self.eval_fct(board, player_id), NO_MOVE

        best_move = None
        best_score = self.min_score if player_id == current_player_id else self.max_score
        for move in board.available_moves:
            new_board = board.play(current_player_id, move)
            winner = new_board.winner
            if winner != EMPTY:
                return self._win_score(winner, current_player_id), move

            score, _ = self._mini_max(new_board, player_id, next_player(current_player_id), alpha, beta, depth=depth-1)
            if player_id == current_player_id:
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

    @staticmethod
    def _win_score(player_id: int, current_player_id) -> int:
        if player_id == current_player_id:
            return 100
        return -100


# TODO - try negascout and other heuristics


"""
Minimax: evaluation functions
"""


class CountOwnedEvaluation(EvaluationFct):
    def __call__(self, board: Board, player_id: PlayerId) -> float:
        count = board.sub_winners.sum()
        if player_id == CIRCLE:
            count *= -1
        return count


class PriceMapEvaluation(EvaluationFct):
    def __init__(self):
        self.sub_weights = np.array([
            [3, 2, 3],
            [2, 4, 2],
            [3, 2, 3]
        ], dtype=np.float32)
        self.weights = np.zeros(shape=(9, 9))
        for x, y in SUB_COORDINATES:
            self.weights[x*3:(x+1)*3, y*3:(y+1)*3] = self.sub_weights * self.sub_weights[(x, y)]
        self.weights /= np.sum(self.weights)
        self.sub_weights /= np.sum(self.sub_weights)

    def __call__(self, board: Board, player_id: PlayerId) -> float:
        winnings = np.where(board.sub_winners != 2, board.sub_winners, 0)
        if player_id == CIRCLE:
            winnings *= -1
        return (board.as_board_matrix() * self.weights).sum() + (winnings * self.sub_weights).sum()


class CombinedEvaluation(EvaluationFct):
    def __init__(self, *evals):
        self.evals = list(evals)

    def __call__(self, board: Board, player_id: PlayerId) -> float:
        return sum(val(board, player_id) for val in self.evals)


# TODO - add an IA that is interested in the difference in cross vs circle in each quadrant? Not really a good strategy


"""
Monte Carlo Tree Search (MCTS) agent
"""


class GameTree:
    __slots__ = ['board', 'total', 'played', 'children']

    def __init__(self, board: Board):
        self.board = board
        self.total: int = 0
        self.played: int = 0
        # TODO - add bias here - based on Machine Learning
        # TODO - add a value function - based on anything...
        self.children = {move: None for move in board.available_moves}

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


class MCTSAgent(Agent):
    def __init__(self, exploration_factor: float):
        self.exploration_factor = exploration_factor
        self.game_tree: GameTree = None

    def on_end_episode(self):
        self.game_tree = None

    def on_opponent_action(self, move: Move, player_id: PlayerId):
        if self.game_tree is not None:
            self.game_tree = self.game_tree.children.get(move, None)

    def get_action(self, board: Board, player_id: PlayerId) -> Move:
        if self.game_tree is None:
            self.game_tree = GameTree(board)

        scenario_count = 0
        chrono = Chronometer()
        chrono.start()
        while chrono.spent() <= 0.8 * MAX_TURN_TIME:
            self._monte_carlo_tree_search(player_id)
            scenario_count += 1

        debug("scenarios:", scenario_count)
        debug("spent:", chrono.spent())

        move, child = self.game_tree.best()
        self.game_tree = None
        return move

    def _monte_carlo_tree_search(self, player_id: PlayerId):

        # Selection (of a node to be expanded)
        current_player_id = player_id
        node = self.game_tree
        move = None
        nodes = []
        while node is not None and not node.board.is_over():
            nodes.append(node)
            move, node = node.select(player_id == current_player_id, self.exploration_factor)
            current_player_id = next_player(current_player_id)

        # Expansion (of a node without statistics)
        if node is None:
            new_board = nodes[-1].board.play(current_player_id, move)
            node = GameTree(new_board)
            nodes[-1].children[move] = node
            nodes.append(node)
            current_player_id = next_player(current_player_id)

        # Play-out (random action until the end)
        board = node.board.clone()
        while not board.is_over():
            moves = board.available_moves
            board.play_(current_player_id, random.choice(moves))
            current_player_id = next_player(current_player_id)

        # Back-propagation (of the score across the tree)
        score = 100 if board.winner == player_id else -100
        for node in nodes:
            node.add_experience(score)


"""
Monte Carlo Tree Search (MCTS) with evaluation agent
"""


# TODO - a kind of MCTS, in which you open lower level just to check the evaluation function (like AlphaGo Zero)


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


def game_loop(agent: Agent):
    board = Board.empty()
    # TODO - recognize the start player instead of considering the opponent is CIRCLE always
    while True:
        opponent_move = read_coord()
        if opponent_move != NO_MOVE:
            board.play_(CIRCLE, opponent_move)
            agent.on_opponent_action(opponent_move, CIRCLE)

        valid_moves = read_valid()
        check_available_moves(valid_moves, board)

        move = agent.get_action(board, CROSS)
        board.play_(CROSS, move)
        print_move(move)
    agent.on_end_episode()


if __name__ == '__main__':
    # game_loop(agent=MinimaxAgent(max_depth=3, eval_fct=CountOwnedEvaluation()))
    game_loop(agent=MinimaxAgent(max_depth=3, eval_fct=PriceMapEvaluation()))
    # game_loop(agent=MCTSAgent(exploration_factor=1.0))
