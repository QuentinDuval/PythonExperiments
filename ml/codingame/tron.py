import sys
import numpy as np
from collections import deque
from typing import List, NamedTuple, Tuple
import time


"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


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


ALL_DIRECTIONS = [move.direction for move in ALL_MOVES]


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
        return World(grid=np.copy(self.grid))

    def remove_player(self, player_id: int):
        debug("REMOVE PLAYER:", player_id)
        self.grid[self.grid == player_id] = EMPTY

    def valid_moves(self, position: Vector) -> List[Move]:
        moves = []
        for move in ALL_MOVES:
            next_position = move.apply(position)
            if self.is_free(next_position):
                moves.append(move)
        return moves

    def is_free(self, position: Vector) -> bool:
        x, y = position
        return 0 <= x < WIDTH and 0 <= y < HEIGHT and self.grid[(x, y)] == EMPTY

    def acquire(self, player_id: int, position: Vector):
        x, y = position
        self.grid[(x, y)] = player_id

    def flood_fill(self, positions: List[Vector], from_player: int) -> ScoresByPlayer:
        scores = np.zeros(shape=len(positions))
        filled = np.copy(self.grid)
        if from_player > 0:
            positions = positions[from_player:] + positions[:from_player]

        to_visit = deque([(x, y) for x, y in positions if (x, y) != (-1, -1)])
        while to_visit:
            x, y = to_visit.popleft()
            owner = filled[(x, y)]
            scores[owner] += 1
            if owner == -1:
                continue

            for direction in ALL_DIRECTIONS:
                next_pos = x + direction[0], y + direction[1]
                if 0 <= next_pos[0] < WIDTH and 0 <= next_pos[1] < HEIGHT:
                    if filled[next_pos] == EMPTY:
                        filled[next_pos] = owner
                        scores[owner] += 1  # count a cell more if it has neighbors, it is better to control
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
        if self._is_player_isolated(self.world, positions, player_id):
            debug("STRATEGY: Wall hugging")
            return self._wall_hugging(self.world, positions, player_id)
        else:
            debug("STRATEGY: Flood filling")
            score, move = self._minimax(self.world, positions, player_id, player_id, depth=2)
            return move

    @classmethod
    def _wall_hugging(cls, world: World, positions: List[Vector], player_id: int) -> Move:
        # TODO - choose the move that removes the least number of edges of the graph (most wall)
        # TODO - but! avoid cutting articulation points, and if you have to, choose the biggest remaining component

        best_move = ALL_MOVES[0]
        best_score = (-1, 0)
        curr_pos = positions[player_id]
        for move in world.valid_moves(positions[player_id]):
            next_pos = move.apply(curr_pos)
            world.acquire(player_id, next_pos)
            flood_fill = world.flood_fill([next_pos], from_player=0)[0]
            wall_count = len(ALL_MOVES) - len(world.valid_moves(next_pos))
            score = (flood_fill, wall_count)
            if score > best_score:
                best_score = score
                best_move = move
            world.acquire(EMPTY, next_pos)
        return best_move

    @classmethod
    def _is_player_isolated(cls, world: World, positions: List[Vector], player_id: int) -> bool:
        world = world.clone()
        to_visit = [positions[player_id]]
        while to_visit:
            position = to_visit.pop()
            for move in ALL_MOVES:
                next_position = move.apply(position)
                if world.is_free(next_position):
                    world.acquire(player_id, next_position)
                    to_visit.append(next_position)
                else:
                    for i in range(len(positions)):
                        if i != player_id and np.array_equal(positions[i], next_position):
                            return False
        return True

    @classmethod
    def _minimax(cls,
                 world: World, positions: List[Vector], player_id: int,
                 current_player_id: int, depth: int) -> Tuple[int, Move]:

        # TODO - UNDERSTAND WHY IT FAILS COMPARED TO HASKELL SOLUTION WHICH IS SUPPOSED TO BE IDENTICAL
        # TODO - flood fill will not allow you to detect you crossed an articulation point
        # TODO - flood fill will clearly not help in the end-game (all directions are the same)

        if depth == 0:
            scores = world.flood_fill(positions, from_player=current_player_id)
            debug("scores:", scores)
            return scores[player_id], None

        curr_position = positions[current_player_id]
        valid_moves = world.valid_moves(curr_position)

        # If there are no valid moves: otherwise the bot will avoid killing opponents
        if not valid_moves:
            # debug("EXPLORATION REMOVE PLAYER:", current_player_id, "at", curr_position)
            new_word = world.clone()
            new_word.remove_player(current_player_id)
            next_player_id = cls._next_player(current_player_id, positions)
            return cls._minimax(new_word, positions, player_id, next_player_id, depth - 1)

        best_move = None
        best_score = float('inf') if player_id != current_player_id else float('-inf')
        for move in valid_moves:
            debug("move of", current_player_id, "at", curr_position, "try", move.name)
            next_position = move.apply(curr_position)
            positions[current_player_id] = next_position
            world.acquire(current_player_id, next_position)
            next_player_id = cls._next_player(current_player_id, positions)
            score, _ = cls._minimax(world, positions, player_id, next_player_id, depth-1)
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

    @classmethod
    def _next_player(cls, current_player_id: int, positions: List[Vector]) -> int:
        next_player_id = current_player_id + 1
        if next_player_id >= len(positions):
            next_player_id -= len(positions)
        return next_player_id


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
