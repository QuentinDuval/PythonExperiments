import sys
import numpy as np
from collections import deque
from typing import List, NamedTuple, Tuple


"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Move
"""


Vector = np.ndarray


class Move(NamedTuple):
    name: str
    direction: Vector

    @classmethod
    def create(cls, name: str, dx: int, dy: int):
        return cls(name=name, direction=np.array([dx, dy], dtype=np.int))

    def apply(self, position: Vector):
        return position + self.direction


ALL_MOVES = [
    Move.create("LEFT", -1, 0),
    Move.create("RIGHT", 1, 0),
    Move.create("UP", 0, -1),
    Move.create("DOWN", 0, 1)]


ALL_DIRECTIONS = [(move.direction[0], move.direction[1]) for move in ALL_MOVES]


"""
World grid
"""


WIDTH = 30
HEIGHT = 20
EMPTY = -1


ScoresByPlayer = np.ndarray


class World:
    def __init__(self, grid):
        self.grid = grid

    @classmethod
    def empty(cls):
        grid = np.full(shape=(WIDTH, HEIGHT), fill_value=EMPTY, dtype=np.int)
        return World(grid)

    def clone(self):
        return World(np.copy(self.grid))

    def remove_player(self, player_id: int):
        debug("REMOVE PLAYER:", player_id)
        self.grid[self.grid == player_id] = EMPTY

    def valid_moves(self, position: Vector) -> List[Move]:
        moves = []
        for move in ALL_MOVES:
            x, y = move.apply(position)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if self.grid[(x, y)] == EMPTY:
                    moves.append(move)
        return moves

    def acquire(self, player_id: int, position: Vector):
        x, y = position
        self.grid[(x, y)] = player_id

    def flood_fill(self, positions: List[Vector], from_player: int) -> ScoresByPlayer:
        scores = np.zeros(shape=len(positions))
        filled = np.copy(self.grid)
        if from_player > 0:
            positions = positions[from_player:] + positions[:from_player]

        to_visit = deque([(x, y) for x, y in positions if filled[(x, y)] != EMPTY])
        while to_visit:
            x, y = to_visit.popleft()
            owner = filled[(x, y)]
            if owner == -1:
                continue

            for dx, dy in ALL_DIRECTIONS:
                next_pos = x + dx, y + dy
                if 0 <= next_pos[0] < WIDTH and 0 <= next_pos[1] < HEIGHT:
                    if filled[next_pos] == EMPTY:
                        filled[next_pos] = owner
                        scores[owner] += 1
                        to_visit.append(next_pos)
        return scores


"""
Agent
"""


class Agent:
    def __init__(self, world: World):
        self.world = world
        self.prediction = None

    def get_action(self, positions: List[Vector], player_id: int) -> Move:
        player_pos = positions[player_id]
        self._check_prediction(player_pos)
        score, move = self._minimax(self.world.clone(), positions, player_id, player_id, depth=3)
        return move

    def _minimax(self, world: World, positions: List[Vector], player_id: int, current_player_id: int, depth: int) -> Tuple[int, Move]:
        # TODO - flood fill will not allow you to detect you crossed an articulation point
        # TODO - flood fill will clearly not help in the end-game (all directions are the same)

        if depth == 0:
            scores = world.flood_fill(positions, from_player=current_player_id)
            return scores[player_id], None

        curr_position = positions[current_player_id]
        valid_moves = self.world.valid_moves(curr_position)

        # If there are no valid moves: otherwise the bot will avoid killing opponents
        if not valid_moves:
            debug("EXPLORATION REMOVE PLAYER:", current_player_id, "at", curr_position)
            new_word = world.clone()
            new_word.remove_player(current_player_id)
            next_player_id = self._next_player(current_player_id, positions)
            return self._minimax(new_word, positions, player_id, next_player_id, depth - 1)

        best_move = None
        best_score = float('inf') if player_id != current_player_id else float('-inf')
        for move in valid_moves:
            next_position = move.apply(curr_position)
            positions[current_player_id] = next_position
            world.acquire(current_player_id, next_position)
            next_player_id = self._next_player(current_player_id, positions)
            score, _ = self._minimax(world, positions, player_id, next_player_id, depth-1)
            world.acquire(EMPTY, next_position)
            positions[current_player_id] = curr_position
            if current_player_id == player_id:
                if score > best_score:
                    best_move = move
                    best_score = score
            else:
                if score < best_score:
                    best_move = move
                    best_score = score
        return best_score, best_move

    def _next_player(self, current_player_id: int, positions: List[Vector]) -> int:
        next_player_id = current_player_id + 1
        if next_player_id >= len(positions):
            next_player_id -= len(positions)
        return next_player_id

    def _check_prediction(self, player_pos: Vector):
        if self.prediction is not None and not np.array_equal(self.prediction, player_pos):
            debug("BAD PREDICTION:", prediction, "vs actual", player_pos)


"""
Input acquisition
"""


class Inputs(NamedTuple):
    nb_player: int
    player_index: int
    init_pos: List[Vector]
    curr_pos: List[Vector]

    def is_over(self, player_id: int) -> bool:
        x, y = self.curr_pos[player_id]
        return x == -1 or y == -1

    @classmethod
    def read(cls):
        nb_player, player_index = [int(i) for i in input().split()]
        init_pos = []
        curr_pos = []
        for i in range(nb_player):
            x0, y0, x1, y1 = [int(j) for j in input().split()]
            init_pos.append(np.array([x0, y0]))
            curr_pos.append(np.array([x1, y1]))
        return cls(nb_player=nb_player, player_index=player_index, init_pos=init_pos, curr_pos=curr_pos)


"""
Game loop
"""


# TODO - high level AI: beginning of game, loosing game, end of game (each player has own connected component), etc.
# TODO - beware of suicidal tendencies of minimax
# TODO - better evaluation that choosing your best score: you need to penalize the opponent as well (pick the higher)


global_world = World.empty()
agent = Agent(global_world)

prev_inputs = None
prediction = None

while True:
    inputs = Inputs.read()
    if prev_inputs:
        for player_id in range(inputs.nb_player):
            if inputs.is_over(player_id):
                if not prev_inputs.is_over(player_id):
                    global_world.remove_player(player_id)
            else:
                global_world.acquire(player_id, inputs.init_pos[player_id])
                global_world.acquire(player_id, inputs.curr_pos[player_id])

    debug(inputs)
    action = agent.get_action(inputs.curr_pos, inputs.player_index)
    print(action.name)
    prev_inputs = inputs
