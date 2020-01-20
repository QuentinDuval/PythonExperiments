from dataclasses import *
import math
import sys
import itertools
from typing import *
import time

import numpy as np

"""
------------------------------------------------------------------------------------------------------------------------
UTILS
------------------------------------------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------------------------------------------
MAIN DATA STRUCTURE
------------------------------------------------------------------------------------------------------------------------
"""


WIDTH = 16000
HEIGHT = 9000


TEAM_CORNERS = np.array([[0, 0], [WIDTH, HEIGHT]], dtype=np.float32)


@dataclass(frozen=False)
class Entities:
    my_team: int
    busters_per_player: int
    position: np.ndarray
    entity_type: np.ndarray     # entity_type: the team id if it is a buster, -1 if it is a ghost.
    carrying: np.ndarray        # state: For busters: 0=idle, 1=carrying a ghost.
    value: np.ndarray           # For busters: Ghost id being carried. For ghosts: number of busters attempting to trap this ghost.

    @property
    def his_team(self):
        return 1 - self.my_team

    @classmethod
    def empty(cls, my_team: int, busters_per_player: int, ghost_count: int):
        size = busters_per_player*2 + ghost_count
        return Entities(
            my_team=my_team,
            busters_per_player=busters_per_player,
            position=np.zeros(shape=(size, 2), dtype=np.float32),
            entity_type=np.zeros(shape=size, dtype=np.int8),
            carrying=np.zeros(shape=size, dtype=bool),
            value=np.zeros(shape=size, dtype=np.int8)
        )


"""
------------------------------------------------------------------------------------------------------------------------
AGENT
------------------------------------------------------------------------------------------------------------------------
"""


class Agent:
    def get_actions(self, entities: Entities) -> List[str]:
        for _ in range(entities.busters_per_player):
            # MOVE x y | BUST id | RELEASE
            yield "MOVE 8000 4500"



"""
------------------------------------------------------------------------------------------------------------------------
GAME LOOP
------------------------------------------------------------------------------------------------------------------------
"""


def read_entities(entities: Entities):
    n = int(input())
    for i in range(n):
        entity_id, x, y, entity_type, state, value = [int(j) for j in input().split()]
        entities.position[entity_id][0] = x
        entities.position[entity_id][1] = y
        entities.entity_type[entity_id] = entity_type
        entities.carrying[entity_id] = state
        entities.value[entity_id] = value


def game_loop():
    busters_per_player = int(input())
    ghost_count = int(input())
    my_team_id = int(input())

    debug("buster by player: ", busters_per_player)
    debug("ghost count: ", ghost_count)
    debug("my team id: ", my_team_id)

    entities = Entities.empty(my_team=my_team_id, busters_per_player=busters_per_player, ghost_count=ghost_count)
    agent = Agent()
    while True:
        read_entities(entities)
        for action in agent.get_actions(entities):
            print(action)


if __name__ == '__main__':
    game_loop()
