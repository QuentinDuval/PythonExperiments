import sys
import numpy as np
from typing import List, NamedTuple, Tuple

"""
Utils
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
World grid
"""

WIDTH = 30
HEIGHT = 20
EMPTY = -1

Vector = np.ndarray


class Move(NamedTuple):
    name: str
    direction: Vector

    @classmethod
    def create(cls, name: str, dx: int, dy: int):
        return cls(name=name, direction=np.array([dx, dy], dtype=np.int))

    def apply(self, position: Vector):
        return position + self.direction


class World:
    def __init__(self):
        self.grid = np.full(shape=(WIDTH, HEIGHT), fill_value=EMPTY, dtype=np.int)
        self.moves = [
            Move.create("LEFT", -1, 0),
            Move.create("RIGHT", 1, 0),
            Move.create("UP", 0, 1),
            Move.create("DOWN", 0, -1)]

    def remove_player(self, player_id: int):
        debug("REMOVE PLAYER:", player_id)
        self.grid[self.grid == player_id] = EMPTY

    def valid_moves(self, position: Vector) -> List[Move]:
        moves = []
        for move in self.moves:
            x, y = move.apply(position)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                debug(self.grid[(x, y)])
                if self.grid[(x, y)] == EMPTY:
                    moves.append(move)
        return moves

    def acquire(self, player_id: int, position: Vector):
        x, y = position
        self.grid[(x, y)] = player_id

    def play(self, player_id: int, position: Vector, move: Move):
        x, y = move.apply(position)
        self.grid[(x, y)] = player_id


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

world = World()

prev_inputs = None
while True:
    inputs = Inputs.read()
    if prev_inputs:
        for player_id in range(inputs.nb_player):
            if inputs.is_over(player_id):
                if not prev_inputs.is_over(player_id):
                    world.remove_player(player_id)
            else:
                world.acquire(player_id, inputs.curr_pos[player_id])

    debug(inputs)
    debug(world.grid)

    player_pos = inputs.curr_pos[inputs.player_index]
    valid_moves = world.valid_moves(player_pos)
    debug(valid_moves)

    for move in valid_moves:
        print(move.name)
        break
    prev_inputs = inputs
