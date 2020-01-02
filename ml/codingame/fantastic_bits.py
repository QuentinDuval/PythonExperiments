"""
Grab Snaffles and try to throw them through the opponent's goal!
Move towards a Snaffle and use your team id to determine where you need to throw it.
"""


from collections import *
from dataclasses import *
import numpy as np
from typing import *
import sys
import math


"""
Utilities
"""


def debug(*args):
    print(*args, file=sys.stderr)


"""
Input acquisition
"""


@dataclass()
class PlayerStatus:
    score: int
    magic: int


def read_status():
    score, magic = [int(i) for i in input().split()]
    return PlayerStatus(score=score, magic=magic)


Vector = np.ndarray


class Entity(NamedTuple):
    entity_id: int
    entity_type: str
    position: Vector
    speed: Vector
    has_snaffle: bool


def read_entities():
    entities = []
    entity_nb = int(input())
    for i in range(entity_nb):
        entity_id, entity_type, x, y, vx, vy, state = input().split()
        entity = Entity(
            entity_id=int(entity_id),
            entity_type=entity_type,
            position=np.array([int(x), int(y)]),
            speed=np.array([int(vx), int(vy)]),
            has_snaffle=int(state) > 0)
        entities.append(entity)
    return entities


"""
Game loop
"""


def game_loop(my_team_id: int):
    debug("my team id:", my_team_id)
    while True:
        player_status = read_status()
        opponent_status = read_status()
        entities = read_entities()

        debug("player status:", player_status)
        debug("opponent status:", opponent_status)
        debug("entities:", entities)

        for i in range(2):
            # Edit this line to indicate the action for each wizard (0 ≤ thrust ≤ 150, 0 ≤ power ≤ 500)
            # i.e.: "MOVE x y thrust" or "THROW x y power"
            print("MOVE 8000 3750 100")


if __name__ == '__main__':
    my_team_id = int(input())  # if 0 you need to score on the right of the map, if 1 you need to score on the left
    game_loop(my_team_id)
